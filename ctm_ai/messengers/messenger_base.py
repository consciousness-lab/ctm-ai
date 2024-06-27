from typing import Any, Callable, Dict, Type, TypeVar

# This TypeVar is used for methods that might need to return or work with instances of subclasses of BaseMessenger.
T = TypeVar('T')


"""
class BaseMessenger(object):
    _messenger_registry: Dict[str, Type["BaseMessenger"]] = {}

    @classmethod
    def register_messenger(
        cls, name: str
    ) -> Callable[[Type["BaseMessenger"]], Type["BaseMessenger"]]:
        def decorator(
            subclass: Type["BaseMessenger"],
        ) -> Type["BaseMessenger"]:
            cls._messenger_registry[name] = subclass
            return subclass

        return decorator

    def __new__(
        cls: Type["BaseMessenger"],
        messenger_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> "BaseMessenger":
        if messenger_name not in cls._messenger_registry:
            raise ValueError(
                f"No messenger registered with name '{messenger_name}'"
            )
        return super(BaseMessenger, cls).__new__(
            cls._messenger_registry[messenger_name]
        )

    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    ) -> None:
        self.init_messenger(role, content)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    ) -> None:
        self.messages: Union[
            str, List[Dict[str, Union[str, Dict[str, Any], List[Any]]]]
        ] = []
        raise NotImplementedError(
            "The 'init_messenger' method must be implemented in derived classes."
        )

    def update_message(
        self, role: str, content: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        raise NotImplementedError(
            "The 'update_message' method must be implemented in derived classes."
        )

    def check_iter_round_num(self) -> int:
        return len(self.messages)

    def add_system_message(
        self, message: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        self.update_message("system", message)

    def add_assistant_message(
        self, message: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        self.update_message("assistant", message)

    def add_user_message(
        self, message: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        self.update_message("user", message)

    def add_user_image(self, image_base64: str) -> None:
        self.update_message(
            "user",
            {
                "type": "image",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            },
        )

    def add_feedback(
        self, feedback: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        self.update_message("system", feedback)

    def clear(self) -> None:
        raise NotImplementedError(
            "The 'clear' method must be implemented in derived classes."
        )

    def get_messages(
        self,
    ) -> Any:
        return self.messages
"""


class BaseMessenger(object):
    _messenger_registry: Dict[str, Type['BaseMessenger']] = {}

    @classmethod
    def register_messenger(
        cls, messenger_name: str
    ) -> Callable[[Type['BaseMessenger']], Type['BaseMessenger']]:
        def decorator(
            subclass: Type['BaseMessenger'],
        ) -> Type['BaseMessenger']:
            cls._messenger_registry[messenger_name] = subclass
            return subclass

        return decorator

    def __new__(cls, messenger_name: str, *args: Any, **kwargs: Any) -> 'BaseMessenger':
        if messenger_name not in cls._messenger_registry:
            raise ValueError(f"No messenger registered with name '{messenger_name}'")
        instance = super(BaseMessenger, cls).__new__(
            cls._messenger_registry[messenger_name]
        )
        instance.name = messenger_name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_messenger(*args, **kwargs)

    def init_messenger(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The 'init_messenger' method must be implemented in derived classes."
        )
