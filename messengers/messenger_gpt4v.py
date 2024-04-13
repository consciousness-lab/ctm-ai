from typing import Dict, List, Union

from messengers.messenger_base import BaseMessenger


@BaseMessenger.register_messenger("gpt4v_messenger")
class GPT4VMessenger(BaseMessenger):
    def __init__(self, role=None, content=None, *args, **kwargs):
        self.init_messenger(role, content)

    def init_messenger(
        self, role: str = None, content: Union[str, Dict, List] = None
    ):
        self.messages = []
        if content and role:
            self.update_messages(role, content)

    def update_message(self, role: str, content: Union[str, Dict, List]):
        self.messages.append({"role": role, "content": content})

    def check_iter_round_num(self):
        return len(self.messages)
