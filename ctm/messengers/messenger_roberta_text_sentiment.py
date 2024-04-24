from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T", bound="BaseMessenger")

from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger("roberta_text_sentiment_messenger")
class RobertaTextSentimentMessenger(BaseMessenger[T]):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(role, content, *args, **kwargs)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    ) -> None:
        self.messages: List[
            Tuple[str, Union[str, Dict[str, Any], List[Any]]]
        ] = []
        if content is not None and role is not None:
            self.update_message(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        self.messages.append((role, content))

    def check_iter_round_num(self) -> int:
        return len(self.messages)
