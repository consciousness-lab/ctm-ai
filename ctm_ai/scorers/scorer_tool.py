from typing import Any, List

from openai import OpenAI

from ..messengers import Message
from ..utils import logprobs_to_softmax, score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer("tool_scorer")
class ToolScorer(BaseScorer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_scorer()

    def init_scorer(self) -> None:
        self.scorer = OpenAI()

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message]) -> float:
        query = messages[-1].query
        gist = messages[-1].gist
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
