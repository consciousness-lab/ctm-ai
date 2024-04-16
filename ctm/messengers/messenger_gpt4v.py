from typing import Any, Dict, List, Optional, Union

from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger("gpt4v_messenger")  # type: ignore[no-untyped-call] # FIX ME
class GPT4VMessenger(BaseMessenger):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict, List]] = None,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.init_messenger(role, content)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict, List]] = None,
    ):
        if role is not None and content is not None:
            self.update_message(role, content)

    def update_message(self, role: str, content: Union[str, Dict, List]):
        self.messages.append({"role": role, "content": content})

    def check_iter_round_num(self) -> int:
        return len(self.messages)
