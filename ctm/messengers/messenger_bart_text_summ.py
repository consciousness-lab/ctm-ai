from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T", bound="BaseMessenger")

from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger("bart_text_summ_messenger")
class BartTextSummarizationMessenger(BaseMessenger[T]):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init_messenger(role, content)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    ) -> None:
        self.messages: List[
            Tuple[str, Union[str, Dict[str, Any], List[Any]]]
        ] = []
        if role and content:
            self.update_message(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        self.messages.append((role, content))

    def check_iter_round_num(self) -> int:
        return len(self.messages)
