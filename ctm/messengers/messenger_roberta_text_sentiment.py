from typing import Callable, Dict, List, Optional, Type, TypeVar, Union

T = TypeVar("T", bound="BaseMessenger")

from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger("roberta_text_sentiment_messenger")
class RobertaTextSentimentMessenger(BaseMessenger):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Union[str, Dict, List] = None,
        *args,
        **kwargs
    ) -> None:
        self.init_messenger(role, content)

    def init_messenger(
        self, role: str = None, content: Union[str, Dict, List] = None
    ) -> None:
        self.messages: str = ""
        if content and role:
            self.update_messages(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict, List]
    ) -> None:
        self.messages = content

    def check_iter_round_num(self) -> int:
        return 1 if len(self.messages) > 0 else 0
