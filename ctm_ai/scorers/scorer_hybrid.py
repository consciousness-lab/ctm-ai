from typing import Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from ..messengers import Message
from ..utils import score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer("hybrid_scorer")
class HybridRelevanceScorer(BaseScorer):
    def init_scorer(self) -> None:
        self.client = OpenAI()

    def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        response = self.client.embeddings.create(model=model, input=[text])
        return np.array(response.data[0].embedding, dtype=np.float32)

    def get_directness_score(self, query: str, gist: str) -> float:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n"
                        f"Answer: {gist}\n\n"
                        "Evaluate how relevant does the answer respond to the question?.\n"
                        "Give a score from 0 to 10 based on the following criteria:\n"
                        "- 10 = Fully answers the question clearly and directly.\n"
                        "- 5 = Partially answers the question, but vague or lacks detail.\n"
                        "- 0 = Does not answer the question or is completely off-topic.\n"
                        "Respond with a single number only. Do not explain."
                    ),
                }
            ],
            max_tokens=10,
        )

        score_text = response.choices[0].message.content.strip()
        print(f"[Directness GPT Response] {score_text}")
        try:
            score = float(score_text)
            return min(max(score / 10.0, 0.0), 1.0)
        except (ValueError, TypeError):
            return 0.0

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message]) -> float:
        query = messages[-1].query
        gist = messages[-1].gist

        if not query or not gist:
            return 0.0

        try:
            query_emb = self.get_embedding(query)
            gist_emb = self.get_embedding(gist)
            topical_score = cosine_similarity([query_emb], [gist_emb])[0][0]
        except Exception as e:
            print(f"[Embedding Error] {e}")
            topical_score = 0.0

        try:
            directness_score = self.get_directness_score(query, gist)
        except Exception as e:
            print(f"[Directness GPT Error] {e}")
            directness_score = 0.0

        # TODO: Try to get the best ratio of topical_score and directness_score
        final_score = 0.3 * topical_score + 0.7 * directness_score
        return float(np.clip(final_score, 0.0, 1.0))
