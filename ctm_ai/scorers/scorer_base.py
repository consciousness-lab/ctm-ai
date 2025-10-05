import math
from typing import Any, Dict, List

import numpy as np
from litellm import completion, embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import word_frequency

from ..utils import configure_litellm, score_exponential_backoff


class BaseScorer(object):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_scorer(*args, **kwargs)

    def init_scorer(self, *args: Any, **kwargs: Any) -> None:
        self.model_name = kwargs.get('model', 'gemini/gemini-2.0-flash-lite')
        self.embedding_model = kwargs.get('embedding_model', 'text-embedding-3-small')
        self.relevance_model = kwargs.get('relevance_model', self.model_name)
        self.confidence_model = kwargs.get('confidence_model', self.model_name)
        self.surprise_model = kwargs.get('scorer_model', self.model_name)

        configure_litellm(model_name=self.model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            response = embedding(model=self.embedding_model, input=[text])
            return np.array(response.data[0]['embedding'], dtype=np.float32)
        except Exception as e:
            print(f'Error getting embedding: {e}')
            return np.zeros(1536, dtype=np.float32)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(
        self, query: str, messages: Dict[str, Any], use_llm: bool = True
    ) -> float:
        query = query
        gist = messages['response']

        if use_llm:
            final_relevance = self._ask_llm_relevance(query, gist)
        else:
            final_relevance = self._ask_statistical_relevance(query, gist)

        return float(np.clip(final_relevance, 0.0, 1.0))

    def _ask_llm_relevance(self, query: str, gist: str) -> float:
        relevance_prompt = [
            {
                'role': 'user',
                'content': """Please evaluate how relevant the answer is to the question on a scale from 0.0 to 1.0.

Question: {query}
Answer: {gist}

Consider:
- 1.0 = Perfectly relevant, directly answers the question with specific information
- 0.8 = Highly relevant, mostly answers the question with useful information
- 0.6 = Moderately relevant, partially answers the question
- 0.4 = Somewhat relevant, tangentially related but not very helpful
- 0.2 = Barely relevant, weak connection or very general response
- 0.0 = Not relevant, refuses to answer, says "cannot determine", or completely unrelated

IMPORTANT: If the answer says "I cannot determine", "I don't know", "cannot answer", or refuses to provide information, score it as 0.0.

Respond with only a number between 0.0 and 1.0 (e.g., 0.85).""",
            }
        ]

        try:
            responses = completion(
                messages=relevance_prompt,
                model=self.relevance_model,
                max_tokens=10,
                temperature=0.2,
                n=1,
            )

            score_text = responses.choices[0].message.content.strip()

            import re

            number_match = re.search(r'(\d+\.?\d*)', score_text)
            if number_match:
                score = float(number_match.group(1))
            else:
                score = float(score_text)

            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            raise ValueError('Error getting relevance score')

    def _ask_statistical_relevance(self, query: str, gist: str) -> float:
        try:
            query_emb = self.get_embedding(query)
            gist_emb = self.get_embedding(gist)

            if np.allclose(query_emb, 0.0) or np.allclose(gist_emb, 0.0):
                topical_score = 0.0
            else:
                topical_score = cosine_similarity([query_emb], [gist_emb])[0][0]
        except Exception as e:
            print(f'[Embedding Error] {e}')
            topical_score = 0.0
        return float(topical_score)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_confidence(self, messages: Dict[str, Any], use_llm: bool = True) -> float:
        gist = messages['response']

        if use_llm:
            final_confidence = self._ask_llm_confidence(gist)
        else:
            final_confidence = self._ask_statistical_confidence(gist)

        return float(np.clip(final_confidence, 0.0, 1.0))

    def _ask_llm_confidence(self, gist: str) -> float:
        confidence_prompt = [
            {
                'role': 'user',
                'content': f"""Please evaluate how confident this response appears to be on a scale from 0.0 to 1.0.

Response: {gist}

Consider:
- 1.0 = Very confident, definitive statements, clear and certain
- 0.8 = Confident, mostly certain with minor qualifications
- 0.6 = Moderately confident, some uncertainty expressed
- 0.4 = Somewhat uncertain, many qualifications or hedging
- 0.2 = Very uncertain, lots of "maybe", "possibly", "might be"
- 0.0 = Completely uncertain, says "cannot determine", "I don't know", or no definitive information

IMPORTANT: If the response says "I cannot determine", "cannot answer", "I don't know", or refuses to provide information, score it as 0.0.

Respond with only a number between 0.0 and 1.0 (e.g., 0.75).""",
            }
        ]

        try:
            responses = completion(
                messages=confidence_prompt,
                model=self.confidence_model,
                max_tokens=10,
                temperature=0.2,
                n=1,
            )

            score_text = responses.choices[0].message.content.strip()

            import re

            number_match = re.search(r'(\d+\.?\d*)', score_text)
            if number_match:
                score = float(number_match.group(1))
            else:
                score = float(score_text)

            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            return 0.5

    def _ask_statistical_confidence(self, gists: List[str]) -> float:
        if len(gists) <= 1:
            return 1.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(gists)
            cos_sim_matrix = cosine_similarity(tfidf_matrix)

            upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
            upper_triangle_values = cos_sim_matrix[upper_triangle_indices]

            avg_similarity = np.mean(upper_triangle_values)

            return float(avg_similarity)

        except Exception:
            return 0.5

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_surprise(
        self,
        query: str,
        messages: Dict[str, Any],
        lang: str = 'en',
        use_llm: bool = True,
    ) -> float:
        gist = messages['response']

        if use_llm:
            final_surprise = self._ask_llm_surprise(query, gist)
        else:
            final_surprise = self._ask_statistical_surprise(gist, lang)

        return float(np.clip(final_surprise, 0.0, 1.0))

    def _ask_llm_surprise(self, query: str, gist: str) -> float:
        """Use LLM to assess how surprising or novel the response is."""
        surprise_prompt = [
            {
                'role': 'user',
                'content': f"""Please evaluate how surprising, unexpected, or novel this response is on a scale from 0.0 to 1.0.

Response: {gist}

Consider:
- 1.0 = Very surprising, highly unexpected, novel insights or information
- 0.8 = Quite surprising, some unexpected elements or perspectives
- 0.6 = Moderately surprising, mix of expected and unexpected content
- 0.4 = Somewhat expected, mostly predictable with minor surprises
- 0.2 = Mostly expected, very predictable content
- 0.0 = Completely expected, entirely predictable, common knowledge

Respond with only a number between 0.0 and 1.0 (e.g., 0.65).""",
            }
        ]

        try:
            responses = completion(
                messages=surprise_prompt,
                model=self.surprise_model,
                max_tokens=10,
                temperature=0.2,
                n=1,
            )

            score_text = responses.choices[0].message.content.strip()

            import re

            number_match = re.search(r'(\d+\.?\d*)', score_text)
            if number_match:
                score = float(number_match.group(1))
            else:
                score = float(score_text)

            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            return 0.5

    def _ask_statistical_surprise(self, gist: str, lang: str) -> float:
        try:
            gist_words = gist.split()
            if not gist_words:
                return 0.0

            log_freqs = [
                -math.log(max(word_frequency(word.lower(), lang), 1e-6))
                for word in gist_words
            ]

            avg_surprise = sum(log_freqs) / len(log_freqs)

            normalized_surprise = avg_surprise / 14.0
            return float(np.clip(normalized_surprise, 0.0, 1.0))

        except Exception:
            return 0.0

    def ask(
        self, query: str, messages: Dict[str, Any], use_llm: bool = True, **kwargs
    ) -> Dict[str, float]:
        relevance = self.ask_relevance(query, messages, use_llm=use_llm)
        confidence = self.ask_confidence(messages, use_llm=use_llm)
        surprise = self.ask_surprise(query, messages, use_llm=use_llm, **kwargs)

        weight = relevance + confidence + surprise

        return {
            'relevance': relevance,
            'confidence': confidence,
            'surprise': surprise,
            'weight': weight,
        }
