from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('vision_processor')
class VisionProcessor(BaseProcessor):
    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='vision_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='vision_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='language_scorer')

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[Union[NDArray[np.float32], str]] = None,
        video_frames: Optional[Union[List[NDArray[np.uint8]], str]] = None,
    ) -> Chunk:
        executor_messages = self.messenger.collect_executor_messages(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        executor_output = self.executor.ask(
            messages=executor_messages,
            image=image,
        )
        scorer_messages = self.messenger.collect_scorer_messages(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
            executor_output=executor_output,
        )
        scorer_output = self.scorer.ask(messages=scorer_messages)

        self.messenger.update(
            executor_output=executor_output,
            scorer_output=scorer_output,
        )

        return self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output
        )
