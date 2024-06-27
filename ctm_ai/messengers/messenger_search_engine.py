from typing import Any, Dict, List, Optional, TypeVar, Union

from .messenger_base import BaseMessenger

T = TypeVar("T", bound="BaseMessenger")


@BaseMessenger.register_messenger("search_engine_messenger")
class SearchEngineMessenger(BaseMessenger):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init_messenger(role, content)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.messages: List[
            Dict[str, Union[str, Dict[str, Any], List[Any]]]
        ] = []
        if role is not None and content is not None:
            self.update_message(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        # Ensuring that 'messages' is defined and typed properly in the base class
        self.messages.append({"role": role, "content": content})

    def check_iter_round_num(self) -> int:
        # Count the number of entries in the messages list
        return len(self.messages)

    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        return query

    def update_executor_messages(self, gist: str):
        return
