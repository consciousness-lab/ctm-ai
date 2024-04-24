from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

T = TypeVar("T", bound="BaseMessenger")

from .messenger_base import BaseMessenger


# If the BaseMessenger has a register_messenger method that is not typed to accept a generic class,
# you might need to define it properly in BaseMessenger or ensure that the typing is correct there.
@BaseMessenger.register_messenger("gpt4v_messenger")
class GPT4VMessenger(BaseMessenger[T]):
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
        if role is not None and content is not None:
            self.update_message(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict[str, Any], List[Any]]
    ):
        # Ensuring that 'messages' is defined and typed properly in the base class
        self.messages.append({"role": role, "content": content})

    def check_iter_round_num(self) -> int:
        # Count the number of entries in the messages list
        return len(self.messages)
