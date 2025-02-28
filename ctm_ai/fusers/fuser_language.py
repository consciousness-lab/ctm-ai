from typing import Any, Dict

from openai import OpenAI

from ..chunks import Chunk
from ..utils import info_exponential_backoff
from .fuser_base import BaseFuser


@BaseFuser.register_fuser('language_fuser')
class LanguageFuser(BaseFuser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_fuser()

    def init_fuser(self) -> None:
        self.model = OpenAI()

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def fuse_info(self, chunk1: Chunk, chunk2: Chunk) -> str | None:
        gist1, gist2 = chunk1.gist, chunk2.gist
        responses = self.model.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'user',
                    'content': f'The following two gists are related and can contain new information when combining them together: 1.{gist1} and 2.{gist2}. Describe the new information generated by combining the two gists:',
                }
            ],
            max_tokens=300,
            n=1,
        )
        answer = (
            responses.choices[0].message.content
            if responses.choices[0].message.content
            else None
        )
        return answer

    def fuse_score(self, chunk1: Chunk, chunk2: Chunk) -> Dict[str, float]:
        relevance = chunk1.relevance + chunk2.relevance
        confidence = chunk1.confidence + chunk2.confidence
        surprise = chunk1.surprise + chunk2.surprise
        weight = relevance * confidence * surprise
        return {
            'relevance': relevance,
            'confidence': confidence,
            'surprise': surprise,
            'weight': weight,
        }
