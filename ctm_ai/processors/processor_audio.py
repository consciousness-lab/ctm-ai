from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('audio_processor')
class AudioProcessor(BaseProcessor):
    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='audio_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='audio_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='language_scorer')

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
    ) -> Chunk:
        return super().ask(
            query,
            text,
            image,
            image_path,
            audio,
            audio_path,
            video_frames,
            video_frames_path,
        )
