from typing import Any, Dict, List

from openai import OpenAI

from ..messengers import BaseMessenger
from ..utils import info_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor("gpt4_executor")
class GPT4Executor(BaseExecutor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.model = OpenAI()

    @info_exponential_backoff()
    def ask(
        self, messages: List[Dict[str, str]], *args: Any, **kwargs: Any
    ) -> str:
        response = self.model.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=300,
            n=1,
        )
        return response.choices[0].message.content
