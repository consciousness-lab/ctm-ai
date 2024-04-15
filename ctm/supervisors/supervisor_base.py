import base64
from typing import Any, Dict, Optional, Type


class BaseSupervisor(object):
    _supervisor_registry: Dict[
        str, Type["BaseSupervisor"]
    ] = {}  # Specify type of keys and values in the dictionary

    @classmethod
    def register_supervisor(
        cls, supervisor_name: str
    ):  # Type annotation for parameter
        def decorator(
            subclass: Type["BaseSupervisor"],
        ) -> Type[
            "BaseSupervisor"
        ]:  # Type annotations for parameters and return type
            cls._supervisor_registry[supervisor_name] = subclass
            return subclass

        return decorator

    def __new__(
        cls, supervisor_name: str, *args, **kwargs
    ) -> Any:  # Type annotation for return type
        if supervisor_name not in cls._supervisor_registry:
            raise ValueError(
                f"No supervisor registered with name '{supervisor_name}'"
            )
        return super(BaseSupervisor, cls).__new__(
            cls._supervisor_registry[supervisor_name]
        )

    def set_model(
        self,
    ) -> None:  # Specify return type None for methods that do not return anything
        raise NotImplementedError(
            "The 'set_model' method must be implemented in derived classes."
        )

    def ask(
        self, query: str, image_path: str
    ) -> (str, float):  # Type annotations for parameters and return type
        gist = self.ask_info(query, image_path)
        score = self.ask_score(query, gist, verbose=True)
        return gist, score

    def ask_info(
        self, query: str, context: Optional[str] = None
    ) -> str:  # Use Optional for parameters that could be None
        raise NotImplementedError(
            "The 'ask_info' method must be implemented in derived classes."
        )  # Updated to raise NotImplementedError

    def ask_score(
        self, query: str, gist: str, verbose: bool = False
    ) -> float:  # Type annotations for parameters and return type
        raise NotImplementedError(
            "The 'ask_score' method must be implemented in derived classes."
        )  # Updated to raise NotImplementedError
