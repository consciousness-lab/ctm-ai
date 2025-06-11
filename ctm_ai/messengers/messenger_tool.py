from typing import List, TypeVar
from .message import Message
from .messenger_base import BaseMessenger

T = TypeVar('T', bound='BaseMessenger')

@BaseMessenger.register_messenger("tool_messenger")
class ToolMessenger(BaseMessenger):
    def collect_executor_messages(self, query: str) -> Message:
        return Message(gist=query)

    def collect_scorer_messages(self, *args, **kwargs) -> List[Message]:
        return self.executor_messages
