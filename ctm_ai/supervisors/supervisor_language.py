from typing import Any, Optional

from litellm import completion

from ..utils import info_exponential_backoff, score_exponential_backoff
from .supervisor_base import BaseSupervisor


@BaseSupervisor.register_supervisor("language_supervisor")
class LanguageSupervisor(BaseSupervisor):
    def init_supervisor(self, *args: Any, **kwargs: Any) -> None:
        super().init_supervisor(*args, **kwargs)
        self.model_name = kwargs.get("supervisor_model", "gemini/gemini-2.0-flash-lite")
        self.supervisors_prompt = kwargs.get("supervisors_prompt", "")

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> Optional[str]:
        messages = [
            {
                "role": "system",
                "content": self.supervisors_prompt,
            },
            {
                "role": "user",
                "content": f"The following is detailed information on the topic: {context}. Based on this information, answer the question: {query}. Answer with a straightforward answer.",
            },
        ]

        try:
            responses = completion(
                messages=messages,
                model=self.info_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=5,
            )
            return responses.choices[0].message.content.strip() if responses else None
        except Exception as e:
            print(f"Error in ask_info: {e}")
            return None

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_score(self, query: str, gist: str, *args: Any, **kwargs: Any) -> float:
        if not gist:
            return 0.0

        messages = [
            {
                "role": "user",
                "content": f"""You are given a query and an answer. 
Task: Decide whether the answer confirms that the query is true/positive, false/negative, or uncertain.

Query: {query}
Answer: {gist}

Output rules:
- If the answer clearly confirms the positive case (e.g., "Yes", "scarstic", "The person is being scarsm"), output 2.
- If the answer clearly confirms the negative case (e.g., "No", "Not scarstic", "The person is not being scarsm"), output 0.
- If the answer is uncertain, ambiguous, or does not confirm either, output 1.

Respond with a single number only: 0, 1, or 2.
""",
            }
        ]

        try:
            responses = completion(
                messages=messages,
                model=self.score_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=1,
            )

            if responses and responses.choices:
                try:
                    # Extract numerical score
                    score_text = responses.choices[0].message.content.strip()
                    score = float(score_text)
                    return score
                    # return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except (ValueError, TypeError):
                    # Fallback to semantic similarity
                    return self._fallback_similarity_score(query, gist)
            else:
                return 0.0

        except Exception as e:
            print(f"Error in ask_score: {e}")
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
