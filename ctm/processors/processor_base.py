from typing import Any, Callable, Dict, Optional, Tuple, Type

from openai import OpenAI

from ..chunks import Chunk
from ..utils.decorator import score_exponential_backoff


class BaseProcessor(object):
    _processor_registry: Dict[str, Type["BaseProcessor"]] = {}

    @classmethod
    def register_processor(
        cls, processor_name: str
    ) -> Callable[[Type["BaseProcessor"]], Type["BaseProcessor"]]:
        def decorator(
            subclass: Type["BaseProcessor"],
        ) -> Type["BaseProcessor"]:
            cls._processor_registry[processor_name] = subclass
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

        executor_prompt = self.messenger.generate_executor_prompt(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )

        gist = self.executor.ask(
            prompt=executor_prompt,
        )

        scorer_prompt = self.messenger.generate_scorer_prompt(
            query=query,
            gist=gist,
        )

        relavance, confidence, surprise, weight = self.scorer.ask(
            prompt=scorer_prompt,
            verbose=True,
        )

        return Chunk(
            processor_name=self.processor_name,
            time_step=0,
            gist=gist,
            relevance=relavance,
            confidence=confidence,
            surprise=surprise,
            weight=weight,
            intensity=weight,
            mood=weight,
        )
