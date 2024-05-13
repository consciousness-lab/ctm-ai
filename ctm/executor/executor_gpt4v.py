from typing import Any

from openai import OpenAI

from ..messengers import BaseMessenger
from ..utils import info_exponential_backoff
from .executor_base import BaseExecutor


class GPT4VExecutor(BaseExecutor):
    def __init__(self) -> None:
        super().__init__()

    def init_model(self) -> None:
        self.model = OpenAI()

    @info_exponential_backoff
    def ask(self, messenger: BaseMessenger, *args: Any, **kwargs: Any) -> str:
        return (
            self.model.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messenger.get_messages(),
                max_tokens=300,
                n=1,
            )
            .choices[0]
            .message.content
        )
