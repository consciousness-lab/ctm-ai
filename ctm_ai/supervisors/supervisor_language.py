from typing import Any, Optional

from ..messengers import Message
from ..utils import (
    ask_llm_standard,
    info_exponential_backoff,
    score_exponential_backoff,
)
from .supervisor_base import BaseSupervisor


@BaseSupervisor.register_supervisor('language_supervisor')
class LanguageSupervisor(BaseSupervisor):
    def init_supervisor(self, *args: Any, **kwargs: Any) -> None:
        super().init_supervisor(*args, **kwargs)
        self.model_name = kwargs.get('supervisor_model', 'gemini/gemini-2.0-flash-lite')
        self.supervisor_system_prompt = kwargs.get(
            'problem_prompt',
            'You are a helpful assistant that can answer questions based on the context. Please provide the answer only based on the detailed information provided.',
        )

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> Optional[str]:
        messages = [
            Message(
                role='system',
                content=self.supervisor_system_prompt,
            ),
            Message(
                role='user',
                content=f'The following is detailed information on the topic: {context}. Based on this information, answer the question: {query}. Answer with a straightforward answer.',
            ),
        ]

        try:
            responses = ask_llm_standard(
                messages=messages,
                model=self.info_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=5,
            )
            return responses[0] if responses else None
        except Exception as e:
            print(f'Error in ask_info: {e}')
            return None

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_score(self, query: str, gist: str, *args: Any, **kwargs: Any) -> float:
        if not gist:
            return 0.0

        messages = [
            Message(
                role='user',
                content=f"""Please evaluate the relevance between the query and the information on a scale from 0.0 to 1.0.

Query: {query}
Information: {gist}

Consider:
- 1.0 = Perfectly relevant, directly answers the question with specific information
- 0.8 = Highly relevant, mostly answers the question with useful information
- 0.6 = Moderately relevant, partially answers the question
- 0.4 = Somewhat relevant, tangentially related but not very helpful
- 0.2 = Barely relevant, weak connection or very general response
- 0.0 = Not relevant, refuses to answer, says "cannot determine", or completely unrelated

IMPORTANT: If the answer says "I cannot determine", "I don't know", "cannot answer", or refuses to provide information, score it as 0.0.

Respond with only a number between 0.0 and 1.0 (e.g., 0.85).""",
            )
        ]

        try:
            responses = ask_llm_standard(
                messages=messages,
                model=self.score_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=1,
            )

            if responses and responses[0]:
                try:
                    # Extract numerical score
                    score_text = responses[0].strip()
                    score = float(score_text)
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except (ValueError, TypeError):
                    # Fallback to semantic similarity
                    return self._fallback_similarity_score(query, gist)
            else:
                return 0.0

        except Exception as e:
            print(f'Error in ask_score: {e}')
            return self._fallback_similarity_score(query, gist)

    def _fallback_similarity_score(self, query: str, gist: str) -> float:
        """Fallback scoring using simple keyword matching."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer()
            docs = [query.lower(), gist.lower()]
            tfidf_matrix = vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(similarity)
        except Exception:
            return 0.5  # Default neutral score
