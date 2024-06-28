from typing import Any, Callable, Dict, List, Type


class BaseMessenger(object):
    _messenger_registry: Dict[str, Type['BaseMessenger']] = {}

    @classmethod
    def register_messenger(
        cls, name: str
    ) -> Callable[[Type['BaseMessenger']], Type['BaseMessenger']]:
        def decorator(
            subclass: Type['BaseMessenger'],
        ) -> Type['BaseMessenger']:
            cls._messenger_registry[name] = subclass
            return subclass

        return decorator

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> 'BaseMessenger':
        if name not in cls._messenger_registry:
            raise ValueError(f"No messenger registered with name '{name}'")
        instance = super(BaseMessenger, cls).__new__(cls._messenger_registry[name])
        instance.name = name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_messenger(*args, **kwargs)

    def check_iter_round_num(self) -> int:
        return len(self.messages)

    def get_executor_messages(self, *args: Any, **kwargs: Any) -> Any:
        return self.messages

    def init_messenger(self, *args: Any, **kwargs: Any) -> None:
        self.messages: List[Any] = []
        raise NotImplementedError(
            "The 'init_messenger' method must be implemented in derived classes."
        )

    def collect_executor_messages(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "The 'collect_executor_messages' method must be implemented in derived classes."
        )

    def update_executor_messages(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "The 'update_executor_messages' method must be implemented in derived classes."
        )
