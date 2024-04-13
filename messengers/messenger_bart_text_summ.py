from typing import Dict, List, Union

from messengers.messenger_base import BaseMessenger


@BaseMessenger.register_messenger("bart_text_summ_messenger")
class BartTextSummarizationMessenger(BaseMessenger):
    def __init__(self, role=None, content=None, *args, **kwargs):
        self.init_messenger(role, content)

    def init_messenger(
        self, role: str = None, content: Union[str, Dict, List] = None
    ):
        self.messages = ""
        if content and role:
            self.update_messages(role, content)

    def update_message(self, role: str, content: Union[str, Dict, List]):
        self.messages += content

    def check_iter_round_num(self):
        return 1 if len(self.messages) > 0 else 0
