from typing import Any, Tuple

from openai import OpenAI

from ..chunks import Chunk
from ..scorers import BaseScorer
from ..utils import info_exponential_backoff, score_exponential_backoff
from .fuser_base import BaseFuser


@BaseFuser.register_fuser("gpt4_fuser")
class GPT4Fuser(BaseFuser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_fuser()

    def init_fuser(self) -> None:
        self.model = OpenAI()
        self.scorer = BaseScorer("gpt4_scorer")

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def fuse_info(self, chunk1: Chunk, chunk2: Chunk) -> Any:
        gist1, gist2 = chunk1.gist, chunk2.gist
        responses = self.model.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"The following two gists are related and can contain new information when combining them together: 1.{gist1} and 2.{gist2}. Describe the new information generated by combining the two gists:",
                }
            ],
            max_tokens=300,
            n=1,
        )
        answer = (
            responses.choices[0].message.content
            if responses.choices[0].message.content
            else "FAILED"
        )
        return answer

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def fuse_score(self, query: str, gist: str) -> Tuple[float, float, float]:
        return self.scorer.ask(query, gist)
