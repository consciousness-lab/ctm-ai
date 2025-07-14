import math
from typing import Any, Callable, Dict, List, Type

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import word_frequency
from litellm import embedding

from ..messengers import Message
from ..utils import ask_llm_standard, configure_litellm, score_exponential_backoff


class BaseScorer(object):
    _scorer_registry: Dict[str, Type["BaseScorer"]] = {}

    @classmethod
    def register_scorer(
        cls, name: str
    ) -> Callable[[Type["BaseScorer"]], Type["BaseScorer"]]:
        def decorator(
            subclass: Type["BaseScorer"],
        ) -> Type["BaseScorer"]:
            cls._scorer_registry[name] = subclass
            return subclass

        return decorator

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> "BaseScorer":
        if name not in cls._scorer_registry:
            raise ValueError(f"No scorer registered with name '{name}'")
        instance = super(BaseScorer, cls).__new__(cls._scorer_registry[name])
        instance.name = name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_scorer(*args, **kwargs)

    def init_scorer(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the scorer with LiteLLM support."""
        # Default configuration for LiteLLM
        self.model_name = kwargs.get("model", "gpt-4o-mini")
        self.embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")
        self.relevance_model = kwargs.get("relevance_model", self.model_name)
        self.confidence_model = kwargs.get("confidence_model", self.model_name)
        self.surprise_model = kwargs.get("surprise_model", self.model_name)

        # Configure LiteLLM
        configure_litellm(model_name=self.model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using LiteLLM."""
        try:
            response = embedding(model=self.embedding_model, input=[text])
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(1536, dtype=np.float32)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message], use_llm: bool = True) -> float:
        """
        Evaluate relevance using LiteLLM.
        Can be overridden by subclasses for specialized relevance scoring.
        """
        if not messages or not messages[-1].query or not messages[-1].gist:
            return 0.0

        query = messages[-1].query
        gist = messages[-1].gist

        llm_relevance = self._ask_llm_relevance(query, gist)
        statistical_relevance = self._ask_statistical_relevance(query, gist)

        if use_llm:
            final_relevance = 0.6 * llm_relevance + 0.4 * statistical_relevance
        else:
            final_relevance = statistical_relevance
        return float(np.clip(final_relevance, 0.0, 1.0))

    def _fallback_relevance_score(self, query: str, gist: str) -> float:
        """Fallback relevance scoring using simple similarity."""
        try:
            vectorizer = TfidfVectorizer()
            docs = [query.lower(), gist.lower()]
            tfidf_matrix = vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def _ask_llm_relevance(self, query: str, gist: str) -> float:

        relevance_prompt = [
            Message(
                role="user",
                content=f"""Please evaluate how relevant the answer is to the question on a scale from 0.0 to 1.0.

Question: {query}
Answer: {gist}

Consider:
- 1.0 = Perfectly relevant, directly answers the question
- 0.8 = Highly relevant, mostly answers the question
- 0.6 = Moderately relevant, partially answers the question
- 0.4 = Somewhat relevant, tangentially related
- 0.2 = Barely relevant, weak connection
- 0.0 = Not relevant, completely unrelated

Respond with only a number between 0.0 and 1.0 (e.g., 0.85).""",
            )
        ]

        try:
            responses = ask_llm_standard(
                messages=relevance_prompt,
                model=self.relevance_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            # Extract numerical score
            score_text = responses[0].strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except (ValueError, TypeError, IndexError):
            return 0.5  # Default neutral confidence

    def _ask_statistical_relevance(self, query: str, gist: str) -> float:
        """Calculate relevance based on consistency of multiple responses."""

        try:
            query_emb = self.get_embedding(query)
            gist_emb = self.get_embedding(gist)

            # Check for zero vectors
            if np.allclose(query_emb, 0.0) or np.allclose(gist_emb, 0.0):
                topical_score = 0.0
            else:
                topical_score = cosine_similarity([query_emb], [gist_emb])[0][0]
        except Exception as e:
            print(f"[Embedding Error] {e}")
            topical_score = 0.0
        return float(topical_score)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_confidence(self, messages: List[Message], use_llm: bool = True) -> float:
        """
        Evaluate confidence using both LLM assessment and statistical methods.
        """
        if not messages or not messages[-1].gist:
            return 0.0

        gist = messages[-1].gist
        gists = messages[-1].gists if messages[-1].gists else [gist]

        # Method 1: LLM-based confidence assessment
        llm_confidence = self._ask_llm_confidence(gist)

        # Method 2: Statistical confidence from multiple responses
        statistical_confidence = self._ask_statistical_confidence(gists)

        # Combine both methods (weighted average)
        if use_llm:
            final_confidence = 0.6 * llm_confidence + 0.4 * statistical_confidence
        else:
            final_confidence = statistical_confidence
        return float(np.clip(final_confidence, 0.0, 1.0))

    def _ask_llm_confidence(self, gist: str) -> float:
        """Use LLM to assess confidence in the response."""
        confidence_prompt = [
            Message(
                role="user",
                content=f"""Please evaluate how confident this response appears to be on a scale from 0.0 to 1.0.

Response: {gist}

Consider:
- 1.0 = Very confident, definitive statements, clear and certain
- 0.8 = Confident, mostly certain with minor qualifications
- 0.6 = Moderately confident, some uncertainty expressed
- 0.4 = Somewhat uncertain, many qualifications or hedging
- 0.2 = Very uncertain, lots of "maybe", "possibly", "might be"
- 0.0 = Completely uncertain, no definitive information

Respond with only a number between 0.0 and 1.0 (e.g., 0.75).""",
            )
        ]

        try:
            responses = ask_llm_standard(
                messages=confidence_prompt,
                model=self.confidence_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            score_text = responses[0].strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            return 0.5  # Default neutral confidence

    def _ask_statistical_confidence(self, gists: List[str]) -> float:
        """Calculate confidence based on consistency of multiple responses."""
        if len(gists) <= 1:
            return 1.0  # Single response, assume full confidence

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(gists)
            cos_sim_matrix = cosine_similarity(tfidf_matrix)

            # Get upper triangle (excluding diagonal)
            upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
            upper_triangle_values = cos_sim_matrix[upper_triangle_indices]

            # Average cosine similarity indicates consistency
            avg_similarity = np.mean(upper_triangle_values)

            return float(avg_similarity)

        except Exception:
            return 0.5  # Default if calculation fails

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_surprise(
        self,
        messages: List[Message],
        lang: str = "en",
        use_llm: bool = True,
    ) -> float:
        """
        Evaluate surprise using both LLM assessment and statistical methods.

        Args:
            messages: List of messages
            lang: Language for word frequency analysis
            use_llm: Whether to use LLM for surprise assessment
        """
        if not messages or not messages[-1].gist:
            return 0.0

        gist = messages[-1].gist

        if use_llm:
            # Method 1: LLM-based surprise assessment
            llm_surprise = self._ask_llm_surprise(gist)

            # Method 2: Statistical surprise from word frequency
            statistical_surprise = self._ask_statistical_surprise(gist, lang)

            # Combine both methods
            final_surprise = 0.7 * llm_surprise + 0.3 * statistical_surprise
            return float(np.clip(final_surprise, 0.0, 1.0))
        else:
            # Use only statistical method
            return self._ask_statistical_surprise(gist, lang)

    def _ask_llm_surprise(self, gist: str) -> float:
        """Use LLM to assess how surprising or novel the response is."""
        surprise_prompt = [
            Message(
                role="user",
                content=f"""Please evaluate how surprising, unexpected, or novel this response is on a scale from 0.0 to 1.0.

Response: {gist}

Consider:
- 1.0 = Very surprising, highly unexpected, novel insights or information
- 0.8 = Quite surprising, some unexpected elements or perspectives
- 0.6 = Moderately surprising, mix of expected and unexpected content
- 0.4 = Somewhat expected, mostly predictable with minor surprises
- 0.2 = Mostly expected, very predictable content
- 0.0 = Completely expected, entirely predictable, common knowledge

Respond with only a number between 0.0 and 1.0 (e.g., 0.65).""",
            )
        ]

        try:
            responses = ask_llm_standard(
                messages=surprise_prompt,
                model=self.surprise_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            score_text = responses[0].strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            return 0.5  # Default neutral surprise

    def _ask_statistical_surprise(self, gist: str, lang: str) -> float:
        """Calculate surprise based on word frequency analysis."""
        try:
            gist_words = gist.split()
            if not gist_words:
                return 0.0

            # Calculate negative log frequencies (higher = more surprising)
            log_freqs = [
                -math.log(max(word_frequency(word.lower(), lang), 1e-6))
                for word in gist_words
            ]

            # Average surprise across all words
            avg_surprise = sum(log_freqs) / len(log_freqs)

            # Normalize to [0, 1] range (14.0 is approximately -log(1e-6))
            normalized_surprise = avg_surprise / 14.0
            return float(np.clip(normalized_surprise, 0.0, 1.0))

        except Exception:
            return 0.0

    def ask(self, messages: List[Message], **kwargs) -> Message:
        """
        Main scoring method that evaluates relevance, confidence, and surprise.

        Returns a Message with all scoring metrics.
        """
        relevance = self.ask_relevance(messages)
        confidence = self.ask_confidence(messages)
        surprise = self.ask_surprise(messages, **kwargs)

        # Calculate composite weight
        weight = relevance * confidence * surprise

        return Message(
            relevance=relevance,
            confidence=confidence,
            surprise=surprise,
            weight=weight,
        )
