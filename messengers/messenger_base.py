from typing import Dict, List, Union


class BaseMessenger(object):
    _messenger_registry = {}  # type: ignore[var-annotated] # FIX ME

    @classmethod
    def register_messenger(cls, messenger_name):  # type: ignore[no-untyped-def] # FIX ME
        def decorator(subclass):  # type: ignore[no-untyped-def] # FIX ME
            cls._messenger_registry[messenger_name] = subclass
            return subclass

        return decorator

    def __new__(cls, messenger_name, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        if messenger_name not in cls._messenger_registry:
            raise ValueError(
                f"No messenger registered with name '{messenger_name}'"
            )
        return super(BaseMessenger, cls).__new__(
            cls._messenger_registry[messenger_name]
        )

    def __init__(self, role=None, content=None, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_messenger(role, content)

    def init_messenger(  # type: ignore[no-untyped-def] # FIX ME
        self, role: str = None, content: Union[str, Dict, List] = None  # type: ignore[assignment, type-arg] # FIX ME
    ):
        pass

    def update_message(self, role: str, content: Union[str, Dict, List]):  # type: ignore[no-untyped-def, type-arg] # FIX ME
        pass

    def check_iter_round_num(self):  # type: ignore[no-untyped-def] # FIX ME
        pass

    def add_system_message(self, message: Union[str, Dict, List]):  # type: ignore[no-untyped-def, type-arg] # FIX ME
        self.update_message("system", message)

    def add_assistant_message(self, message: Union[str, Dict, List]):  # type: ignore[no-untyped-def, type-arg] # FIX ME
        self.update_message("assistant", message)

    def add_user_message(self, message: Union[str, Dict, List]):  # type: ignore[no-untyped-def, type-arg] # FIX ME
        self.update_message("user", message)

    def add_user_image(self, image_base64: str):  # type: ignore[no-untyped-def] # FIX ME
        self.add_message(  # type: ignore[attr-defined] # FIX ME
            "user",
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            },
        )

    def add_feedback(self, feedback: Union[str, Dict, List]):  # type: ignore[no-untyped-def, type-arg] # FIX ME
        self.add_message("system", feedback)  # type: ignore[attr-defined] # FIX ME

    def clear(self):  # type: ignore[no-untyped-def] # FIX ME
        self.messages.clear()  # type: ignore[attr-defined] # FIX ME

    def get_messages(self):  # type: ignore[no-untyped-def] # FIX ME
        return self.messages  # type: ignore[attr-defined] # FIX ME
