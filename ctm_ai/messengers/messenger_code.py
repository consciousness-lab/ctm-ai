from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .message import Message
from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger('code_messenger')
class CodeMessenger(BaseMessenger):
    def collect_executor_messages(
            self,
            query: str,
            text: Optional[str] = None,
            image: Optional[str] = None,
            audio: Optional[Union[NDArray[np.float32], str]] = None,
            video_frames: Optional[Union[List[NDArray[np.uint8]], str]] = None,
    ) -> List[Message]:
        content = 'Query: {}\n'.format(query)
        if text is not None:
            content += 'Text: {}\n'.format(text)
        message = Message(
            role='user',
            content=content,
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
            role='user',
            query=query,
            gist=executor_output.gist,
            gists=executor_output.gists,
        )
        self.scorer_messages.append(message)
        return self.scorer_messages
