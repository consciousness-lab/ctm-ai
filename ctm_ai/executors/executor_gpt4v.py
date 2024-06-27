from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..utils import info_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('gpt4v_executor')
class GPT4VExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.model = OpenAI()

    @info_exponential_backoff()
    def ask(self, messages: list[ChatCompletionMessageParam]) -> str | None:
        response = self.model.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=messages,
            max_tokens=300,
            n=1,
        )
        return response.choices[0].message.content
