from typing import Any, Callable, Dict, Type


class BaseExecutor(object):
    _processor_registry: Dict[str, Type["BaseExecutor"]] = {}

    @classmethod
    def register_processor(
        cls, processor_name: str
    ) -> Callable[[Type["BaseExecutor"]], Type["BaseExecutor"]]:
        def decorator(
            subclass: Type["BaseExecutor"],
        ) -> Type["BaseExecutor"]:
            cls._processor_registry[processor_name] = subclass
            return subclass

        return decorator

    def __new__(
        cls, processor_name: str, *args: Any, **kwargs: Any
    ) -> "BaseExecutor":
        if processor_name not in cls._processor_registry:
            raise ValueError(
                f"No processor registered with name '{processor_name}'"
            )
        return super(BaseExecutor, cls).__new__(
            cls._processor_registry[processor_name]
        )

    def __init__(self) -> None:
        self.init_model()

    def ask(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(
            "The 'execute' method must be implemented in derived classes."
        )
