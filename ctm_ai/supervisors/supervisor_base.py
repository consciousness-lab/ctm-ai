from typing import Any, Dict, Optional, Tuple, Type, Union

from ..utils import logging_ask


class BaseSupervisor(object):
    _supervisor_registry: Dict[str, Type['BaseSupervisor']] = {}

    @classmethod
    def register_supervisor(cls, name: str) -> Any:
        def decorator(
            subclass: Type['BaseSupervisor'],
        ) -> Type['BaseSupervisor']:
            cls._supervisor_registry[name] = subclass
            return subclass

        return decorator

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in cls._supervisor_registry:
            raise ValueError(f"No supervisor registered with name '{name}'")
        instance = super(BaseSupervisor, cls).__new__(cls._supervisor_registry[name])
        instance.name = name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_supervisor()

    def init_supervisor(
        self,
    ) -> None:
        raise NotImplementedError(
            "The 'set_model' method must be implemented in derived classes."
        )

    @logging_ask()
    def ask(self, query: str, image_path: str) -> Tuple[Union[str, None], float]:
        answer = self.ask_answer(query, image_path)
        score = self.ask_score(query, answer)
        return answer, score

    def ask_answer(self, query: str, context: Optional[str] = None) -> str | None:
        raise NotImplementedError(
            "The 'ask_answer' method must be implemented in derived classes."
        )

    def ask_score(self, query: str, gist: str | None) -> float:
        raise NotImplementedError(
            "The 'ask_score' method must be implemented in derived classes."
        )
