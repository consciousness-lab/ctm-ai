from typing import Dict, List, Union

from messengers.messenger_base import BaseMessenger


@BaseMessenger.register_messenger("bart_text_summ_messenger")  # type: ignore[no-untyped-call] # FIX ME
class BartTextSummarizationMessenger(BaseMessenger):
    def __init__(self, role=None, content=None, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_messenger(role, content)

    def init_messenger(  # type: ignore[no-untyped-def] # FIX ME
        self, role: str = None, content: Union[str, Dict, List] = None  # type: ignore[assignment, type-arg] # FIX ME
    ):
        self.messages = ""
        if content and role:
            self.update_messages(role, content)  # type: ignore[attr-defined] # FIX ME

    def update_message(self, role: str, content: Union[str, Dict, List]):  # type: ignore[no-untyped-def, type-arg] # FIX ME
        self.messages += content  # type: ignore[operator] # FIX ME

    def check_iter_round_num(self):  # type: ignore[no-untyped-def] # FIX ME
        return 1 if len(self.messages) > 0 else 0
