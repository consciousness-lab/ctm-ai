from typing import Any, Callable, Dict, Optional, Tuple, Type

from openai import OpenAI

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
        cls, processor_name: str, *args: Any, **kwargs: Any
    ) -> "BaseProcessor":
        if processor_name not in cls._processor_registry:
            raise ValueError(
                f"No processor registered with name '{processor_name}'"
            )
        return super(BaseProcessor, cls).__new__(
            cls._processor_registry[processor_name]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_executor()
        self.init_messenger()
        self.init_scorer()
        self.init_task()

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

    def init_task(self) -> None:
        raise NotImplementedError(
            "The 'init_task_info' method must be implemented in derived classes."
        )

    def ask(
        self, query: str, text: str, image: str, audio: str, video_frames: str
    ) -> str:
        gist = self.ask_info(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        return gist

    def ask_info(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(
            "The 'ask_info' method must be implemented in derived classes."
        )
