from typing import Any, Callable, Dict, Optional, Tuple, Type

from openai import OpenAI

from ..utils.decorator import score_exponential_backoff


class BaseProcessor:
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
        self.init_scorer()
        self.init_processor()
        self.init_messenger()
        self.init_task_info()

    def init_processor(self) -> None:
        raise NotImplementedError(
            "The 'init_processor' method must be implemented in derived classes."
        )

    def init_messenger(self) -> None:
        raise NotImplementedError(
            "The 'init_messenger' method must be implemented in derived classes."
        )

    def init_task_info(self) -> None:
        raise NotImplementedError(
            "The 'init_task_info' method must be implemented in derived classes."
        )

    def init_scorer(self) -> None:
        self.scorer = OpenAI()
        raise NotImplementedError(
            "The 'init_scorer' method must be implemented in derived classes."
        )

    def ask(
        self, query: str, text: str, image: str, audio: str, video_frames: str
    ) -> Tuple[str, float]:
        gist = self.ask_info(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        score = self.ask_score(query, gist, verbose=True)
        return gist, score

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, query: str, gist: str) -> float:
        response = self.scorer.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"How related is the information ({gist}) with the query ({query})? Answer with a number from 0 to 5 and do not add any other thing.",
                }
            ],
            max_tokens=50,
        )
        score = (
            float(response.choices[0].message.content.strip()) / 5
            if response.choices[0].message.content
            else 0.0
        )
        return score

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_confidence(self, query: str, gist: str) -> float:
        response = self.scorer.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"How confident do you think the information ({gist}) is a must-know? Answer with a number from 0 to 5 and do not add any other thing.",
                }
            ],
            max_tokens=50,
        )
        score = (
            float(response.choices[0].message.content.strip()) / 5
            if response.choices[0].message.content
            else 0.0
        )
        return score

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_surprise(
        self, query: str, gist: str, history_gists: Optional[str] = None
    ) -> float:
        response = self.scorer.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"How surprising do you think the information ({gist}) is as an output of the processor? Answer with a number from 0 to 5 and do not add any other thing.",
                }
            ],
            max_tokens=50,
        )
        score = (
            float(response.choices[0].message.content.strip()) / 5
            if response.choices[0].message.content
            else 0.0
        )
        return score

    def ask_score(
        self,
        query: str,
        gist: str,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        relevance = self.ask_relevance(query, gist, *args, **kwargs)
        confidence = self.ask_confidence(query, gist, *args, **kwargs)
        surprise = self.ask_surprise(query, gist, *args, **kwargs)
        if verbose:
            print(
                f"Relevance: {relevance}, Confidence: {confidence}, Surprise: {surprise}"
            )

        final_score = relevance * confidence * surprise
        return final_score

    def ask_info(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(
            "The 'ask_info' method must be implemented in derived classes."
        )
