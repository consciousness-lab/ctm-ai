import base64
from typing import Any, Dict, Optional, Tuple, Type

from ..utils import logging_ask


class BaseSupervisor(object):
    _supervisor_registry: Dict[str, Type["BaseSupervisor"]] = {}

    @classmethod
    def register_supervisor(cls, supervisor_name: str) -> Any:
        def decorator(
            subclass: Type["BaseSupervisor"],
        ) -> Type["BaseSupervisor"]:
            cls._supervisor_registry[supervisor_name] = subclass
            return subclass

        return decorator

    def __new__(cls, supervisor_name: str, *args: Any, **kwargs: Any) -> Any:
        if supervisor_name not in cls._supervisor_registry:
            raise ValueError(
                f"No supervisor registered with name '{supervisor_name}'"
            )
        return super(BaseSupervisor, cls).__new__(
            cls._supervisor_registry[supervisor_name]
        )

    def init_supervisor(
        self,
    ) -> None:
        raise NotImplementedError(
            "The 'set_model' method must be implemented in derived classes."
        )

    @logging_ask()
    def ask(self, query: str, image_path: str) -> Tuple[str, float]:
        gist = self.ask_info(query, image_path)
        score = self.ask_score(query, gist, verbose=True)
        return gist, score

    def ask_info(self, query: str, context: Optional[str] = None) -> str:
        raise NotImplementedError(
            "The 'ask_info' method must be implemented in derived classes."
        )

    def ask_score(self, query: str, gist: str, verbose: bool = False) -> float:
        raise NotImplementedError(
            "The 'ask_score' method must be implemented in derived classes."
        )
