from typing import Any, Callable, Dict, Optional, Type

from openai import OpenAI

from ..utils.decorator import score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer("gpt4_scorer")
class GPT4Scorer(BaseScorer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_scorer()

    def init_scorer(self) -> None:
        self.scorer = OpenAI()

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, query: str, gist: str) -> float:
        response = self.scorer.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"How related is the information ({gist}) with the query ({query})? Answer with a number from 0 to 5 and do not add any other thing.",
                }
            ],
            max_tokens=50,
        )
        score = (
            float(response.choices[0].message.content.strip()) / 5
            if response.choices[0].message.content
            else 0.0
        )
        return score

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_confidence(self, query: str, gist: str) -> float:
        response = self.scorer.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"How confident do you think the information ({gist}) is a must-know? Answer with a number from 0 to 5 and do not add any other thing.",
                }
            ],
            max_tokens=50,
        )
        score = (
            float(response.choices[0].message.content.strip()) / 5
            if response.choices[0].message.content
            else 0.0
        )
        return score

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_surprise(
        self, query: str, gist: str, history_gists: Optional[str] = None
    ) -> float:
        response = self.scorer.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"How surprising do you think the information ({gist}) is as an output of the processor? Answer with a number from 0 to 5 and do not add any other thing.",
                }
            ],
            max_tokens=50,
        )
        score = (
            float(response.choices[0].message.content.strip()) / 5
            if response.choices[0].message.content
            else 0.0
        )
        return score
