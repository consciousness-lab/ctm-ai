from typing import Any, Dict, List, Optional, TypeVar, Union

from .messenger_base import BaseMessenger

T = TypeVar("T", bound="BaseMessenger")


@BaseMessenger.register_messenger("roberta_text_sentiment_messenger")
class RobertaTextSentimentMessenger(BaseMessenger):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(role, content, *args, **kwargs)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    ) -> None:
        self.messages: str = ""
        if role and content:
            self.update_message(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        self.messages += content

    def check_iter_round_num(self) -> int:
        return len(self.messages)
