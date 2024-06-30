from typing import Any, Callable, Dict, Optional, Type

from ..chunks import Chunk
from ..executors import BaseExecutor
from ..messengers import BaseMessenger
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
        self.init_messenger()
        self.init_executor()
        self.init_scorer()

    def init_executor(self) -> None:
        self.executor = BaseExecutor(name='gpt4_executor')
        raise NotImplementedError(
            "The 'init_executor' method must be implemented in derived classes."
        )

    def init_messenger(self) -> None:
        self.messenger = BaseMessenger(name='gpt4_messenger')
        raise NotImplementedError(
            "The 'init_messenger' method must be implemented in derived classes."
        )

    def init_scorer(self) -> None:
        self.scorer = BaseScorer(name='gpt4_scorer')
        raise NotImplementedError(
            "The 'init_scorer' method must be implemented in derived classes."
        )

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        video_frames: Optional[Any] = None,
    ) -> Chunk:
        executor_messages = self.messenger.collect_executor_message(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )

        executor_output = self.executor.ask(messages=executor_messages)

        gist, scorer_inputs = self.messenger.parse_executor_message(
            query=query,
            executor_output=executor_output,
        )

        self.messenger.update_executor_message(gist=gist)

        self.messenger.collect_scorer_message(
            query=query,
            gist=gist,
            executor_output=executor_output,
        )

        score = self.scorer.ask(
            query=query,
            gists=scorer_inputs,
        )
        self.messenger.parse_scorer_message(score=score)
        self.messenger.update_scorer_message(gist=gist)

        return Chunk(
            time_step=0,
            processor_name=self.name,
            gist=gist,
            relevance=score['relevance'],
            confidence=score['confidence'],
            surprise=score['surprise'],
            weight=score['weight'],
            intensity=score['weight'],
            mood=score['weight'],
        )

    def update(self, chunk: Chunk) -> None:
        if chunk.processor_name != self.name:
            self.messenger.update_executor_messages(gist=chunk.gist)
