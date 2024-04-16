from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar(
    "T", bound="BaseMessenger"
)  # For typing class methods and the registry


class BaseMessenger:
    _messenger_registry: Dict[str, Type[T]] = {}

    @classmethod
    def register_messenger(
        cls: Type[T], messenger_name: str
    ) -> Callable[[Type[T]], Type[T]]:
        def decorator(subclass: Type[T]) -> Type[T]:
            cls._messenger_registry[messenger_name] = subclass
            return subclass

        return decorator

    def __new__(
        cls: Type[T], messenger_name: str, *args: Any, **kwargs: Any
    ) -> T:
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
        content: Optional[Union[str, Dict, List]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.init_messenger(role, content)
        self.messages: List[
            Tuple[str, Union[str, Dict, List]]
        ] = []  # Assume messages are stored as a list of tuples

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict, List]] = None,
    ) -> None:
        if role and content:
            self.update_message(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict, List]
    ) -> None:
        self.messages.append((role, content))

    def check_iter_round_num(self) -> int:
        return len(self.messages)

    def add_system_message(self, message: Union[str, Dict, List]) -> None:
        self.update_message("system", message)

    def add_assistant_message(self, message: Union[str, Dict, List]) -> None:
        self.update_message("assistant", message)

    def add_user_message(self, message: Union[str, Dict, List]) -> None:
        self.update_message("user", message)

    def add_user_image(self, image_base64: str) -> None:
        self.update_message(
            "user",
            {
                "type": "image",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            },
        )

    def add_feedback(self, feedback: Union[str, Dict, List]) -> None:
        self.update_message("system", feedback)

    def clear(self) -> None:
        self.messages.clear()

    def get_messages(self) -> List[Tuple[str, Union[str, Dict, List]]]:
        return self.messages
