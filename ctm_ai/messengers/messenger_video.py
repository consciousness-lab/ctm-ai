from typing import List, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from .message import Message
from .messenger_base import BaseMessenger

T = TypeVar('T', bound='BaseMessenger')


@BaseMessenger.register_messenger('video_messenger')
class VideoMessenger(BaseMessenger):
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
    ) -> List[Message]:
        content = 'Query: {}\n'.format(query)
        if text is not None:
            content += 'Text: {}\n'.format(text)
        if (video_frames or video_frames_path) and video_frames_path:
            content += 'Note: The input contains {} video frames. Please integrate visual information across these frames for a comprehensive analysis.\n'.format(
                len(video_frames_path)
            )
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
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
    ) -> List[Message]:
        message = Message(
            role='assistant',
            query=query,
            gist=executor_output.gist,
            gists=executor_output.gists,
        )
        self.scorer_messages.append(message)
        return self.scorer_messages
