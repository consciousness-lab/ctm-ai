from typing import Any, List

from ..messengers import Message
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('language_executor')
class LanguageExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using the base class functionality."""
        super().init_model(*args, **kwargs)

    def ask(
        self,
        messages: List[Message],
        max_token: int = 300,
        return_num: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for language processing using the base class functionality."""
        return self.ask_standard(
            messages=messages,
            max_token=max_token,
            return_num=return_num,
            *args,
            **kwargs,
        )
