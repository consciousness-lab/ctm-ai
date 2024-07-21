from typing import Any, Callable, Dict, Optional, Tuple, Type

from ..chunks import Chunk
from ..executors import BaseExecutor
from ..messengers import BaseMessenger, Message
from ..scorers import BaseScorer


class BaseProcessor(object):
    _processor_registry: Dict[str, Type['BaseProcessor']] = {}

    @classmethod
    def register_processor(
        cls, name: str
    ) -> Callable[[Type['BaseProcessor']], Type['BaseProcessor']]:
        def decorator(
            subclass: Type['BaseProcessor'],
        ) -> Type['BaseProcessor']:
            cls._processor_registry[name] = subclass
            return subclass

        return decorator

    def __new__(
        cls,
        name: str,
        group_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> 'BaseProcessor':
        if name not in cls._processor_registry:
            raise ValueError(f"No processor registered with name '{name}'")
        subclass = cls._processor_registry[name]
        instance = super(BaseProcessor, cls).__new__(subclass)
        instance.name = name
        instance.group_name = group_name
        return instance

    def __init__(
        self, name: str, group_name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        self.name = name
        self.group_name = group_name
        self.alpha = 1.0
        self.executor = self.init_executor()
        self.messenger = self.init_messenger()
        self.scorer = self.init_scorer()

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='gpt4_executor')

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='gpt4_messenger')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='gpt4_scorer')

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        video_frames: Optional[Any] = None,
    ) -> Chunk:
        executor_messages = self.messenger.collect_executor_messages(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )

        executor_output = self.executor.ask(messages=executor_messages)

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

        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            alpha=self.alpha,
        )
        return chunk

    def update(self, chunk: Chunk) -> None:
        if chunk.processor_name == self.name:
            if chunk.feedback is True:
                self.alpha = self.aplha * 2
            elif chunk.feedback is False:
                self.alpha = self.alpha * 0.5

        executor_output, scorer_output = self.split_chunk_into_outputs(chunk)
        self.messenger.update(
            executor_output=executor_output,
            scorer_output=scorer_output,
        )

    def merge_outputs_into_chunk(
        self, name: str, scorer_output: Message, executor_output: Message, alpha: float
    ) -> Chunk:
        return Chunk(
            time_step=0,
            processor_name=name,
            gist=executor_output.content,
            relevance=scorer_output.relevance,
            confidence=scorer_output.confidence,
            surprise=scorer_output.surprise,
            weight=scorer_output.weight * alpha,
            intensity=scorer_output.weight * alpha,
            mood=scorer_output.weight * alpha,
        )

    def split_chunk_into_outputs(self, chunk: Chunk) -> Tuple[Message, Message]:
        executor_output = Message(
            role='assistant',
            content=chunk.gist,
        )
        scorer_output = Message(
            relevance=chunk.relevance,
            confidence=chunk.confidence,
            surprise=chunk.surprise,
            weight=chunk.weight,
        )
        return executor_output, scorer_output
