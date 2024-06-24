from typing import Any, Callable, Dict, Optional, Type

from ..chunks import Chunk


class BaseProcessor(object):
    _processor_registry: Dict[str, Type["BaseProcessor"]] = {}

    @classmethod
    def register_processor(
        cls, name: str
    ) -> Callable[[Type["BaseProcessor"]], Type["BaseProcessor"]]:
        def decorator(
            subclass: Type["BaseProcessor"],
        ) -> Type["BaseProcessor"]:
            cls._processor_registry[name] = subclass
            return subclass

        return decorator

    def __new__(
        cls,
        name: str,
        group_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "BaseProcessor":
        if name not in cls._processor_registry:
            raise ValueError(f"No processor registered with name '{name}'")
        subclass = cls._processor_registry[name]
        instance = super(BaseProcessor, cls).__new__(subclass)
        instance.name = name
        instance.group_name = group_name
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_messenger()
        self.init_executor()
        self.init_scorer()

    def init_executor(self) -> None:
        raise NotImplementedError(
            "The 'init_executor' method must be implemented in derived classes."
        )

    def init_messenger(self) -> None:
        raise NotImplementedError(
            "The 'init_messenger' method must be implemented in derived classes."
        )

    def init_scorer(self) -> None:
        raise NotImplementedError(
            "The 'init_scorer' method must be implemented in derived classes."
        )

    def ask(
        self, query: str, text: str, image: Any, audio: Any, video_frames: Any
    ) -> Chunk:

        executor_messages = self.messenger.collect_executor_messages(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )

        gist = self.executor.ask(messages=executor_messages)

        self.messenger.update_executor_messages(gist=gist)

        score = self.scorer.ask(
            query=query,
            gist=gist,
        )

        return Chunk(
            time_step=0,
            processor_name=self.name,
            gist=gist,
            relevance=score["relevance"],
            confidence=score["confidence"],
            surprise=score["surprise"],
            weight=score["weight"],
            intensity=score["weight"],
            mood=score["weight"],
        )

    def update(self, chunk: Chunk) -> None:
        if chunk.processor_name != self.name:
            self.messenger.update_executor_messages(gist=chunk.gist)
