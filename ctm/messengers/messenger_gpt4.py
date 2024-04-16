from typing import Dict, List, Optional, Union

from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger("gpt4_messenger")  # type: ignore[no-untyped-call] # FIX ME
class GPT4Messenger(BaseMessenger):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict, List]] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            *args, **kwargs
        )  # Ensure proper initialization of base class if necessary
        self.init_messenger(role, content)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict, List]] = None,
    ):
        self.messages: List[
            Dict[str, Union[str, Dict, List]]
        ] = []  # Explicitly type the messages list
        if content is not None and role is not None:
            self.update_message(role, content)

    def update_message(self, role: str, content: Union[str, Dict, List]):
        self.messages.append({"role": role, "content": content})

    def check_iter_round_num(self) -> int:
        return len(self.messages)
