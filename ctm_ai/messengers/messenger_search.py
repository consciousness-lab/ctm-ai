from typing import List, Optional, Any, Union, TypeVar
import numpy as np
from numpy.typing import NDArray

from .message import Message
from .messenger_base import BaseMessenger

T = TypeVar('T', bound='BaseMessenger')


@BaseMessenger.register_messenger('search_messenger')
class SearchMessenger(BaseMessenger):
    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[Union[NDArray[np.float32], str]] = None,
        video_frames: Optional[Union[List[NDArray[np.uint8]], str]] = None,
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
        audio: Optional[Union[NDArray[np.float32], str]] = None,
        video_frames: Optional[Union[List[NDArray[np.uint8]], str]] = None,
    ) -> List[Message]:
        message = Message(
            role='assistant',
            gist=executor_output.gist,
        )
        self.scorer_messages.append(message)
        return self.scorer_messages
