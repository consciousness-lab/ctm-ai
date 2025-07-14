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

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> Optional[str]:
        messages = [
            Message(
                role='user',
                content=f'The following is detailed information on the topic: {context}. Based on this information, answer the question: {query}. Answer with a few words:',
            )
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
- 1.0 = Perfectly relevant, directly answers the query
- 0.8 = Highly relevant, mostly answers the query
- 0.6 = Moderately relevant, partially answers the query
- 0.4 = Somewhat relevant, tangentially related
- 0.2 = Barely relevant, weak connection
- 0.0 = Not relevant, completely unrelated

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
