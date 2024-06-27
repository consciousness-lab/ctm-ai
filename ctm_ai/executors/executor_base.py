from typing import Any, Callable, Dict, Type


class BaseExecutor(object):
    _executor_registry: Dict[str, Type['BaseExecutor']] = {}

    @classmethod
    def register_executor(
        cls, executor_name: str
    ) -> Callable[[Type['BaseExecutor']], Type['BaseExecutor']]:
        def decorator(
            subclass: Type['BaseExecutor'],
        ) -> Type['BaseExecutor']:
            cls._executor_registry[executor_name] = subclass
            return subclass

        return decorator

    def __new__(cls, executor_name: str, *args: Any, **kwargs: Any) -> 'BaseExecutor':
        if executor_name not in cls._executor_registry:
            raise ValueError(f"No executor registered with name '{executor_name}'")
        instance = super(BaseExecutor, cls).__new__(
            cls._executor_registry[executor_name]
        )
        instance.name = executor_name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_model(*args, **kwargs)

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The 'init_model' method must be implemented in derived classes."
        )

    def ask(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "The 'ask' method must be implemented in derived classes."
        )
