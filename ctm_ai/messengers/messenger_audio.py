from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .message import Message
from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger('audio_messenger')
class AudioMessenger(BaseMessenger):
    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
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
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> List[Message]:
        self.scorer_messages = [
            Message(
                role='user',
                query=query,
                gist=executor_output.gist,
                gists=executor_output.gists,
            )
        ]
        return self.scorer_messages
