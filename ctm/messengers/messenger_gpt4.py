from typing import Dict, List, Union

from ctm.messengers.messenger_base import BaseMessenger


@BaseMessenger.register_messenger("gpt4_messenger")  # type: ignore[no-untyped-call] # FIX ME
class GPT4Messenger(BaseMessenger):
    def __init__(self, role=None, content=None, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_messenger(role, content)

    def init_messenger(  # type: ignore[no-untyped-def] # FIX ME
        self, role: str = None, content: Union[str, Dict, List] = None  # type: ignore[assignment, type-arg] # FIX ME
    ):
        self.messages = []  # type: ignore[var-annotated] # FIX ME
        if content and role:
            self.update_messages(role, content)  # type: ignore[attr-defined] # FIX ME

    def update_message(self, role: str, content: Union[str, Dict, List]):  # type: ignore[no-untyped-def, type-arg] # FIX ME
        self.messages.append({"role": role, "content": content})

    def check_iter_round_num(self):  # type: ignore[no-untyped-def] # FIX ME
        return len(self.messages)
