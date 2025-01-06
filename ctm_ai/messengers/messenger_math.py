from typing import List, Optional, TypeVar

from .message import Message
from .messenger_base import BaseMessenger

T = TypeVar('T', bound='BaseMessenger')


@BaseMessenger.register_messenger('math_messenger')
class MathMessenger(BaseMessenger):
    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[str] = None,
    ) -> List[Message]:
        message = Message(
            role='user',
            content=query,
        )
        self.executor_messages.append(message)
        return self.executor_messages

    def collect_scorer_messages(
        self,
        executor_output: Message,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[str] = None,
    ) -> List[Message]:
        message = Message(
            role='assistant',
            gist=executor_output.gist,
        )
        self.scorer_messages.append(message)
        return self.scorer_messages
