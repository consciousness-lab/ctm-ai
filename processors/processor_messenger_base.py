from typing import Union, List, Dict

class BaseProcessorMessenger(object):
    _messenger_registry = {}

    @classmethod
    def register_messenger(cls, messenger_name):
        def decorator(subclass):
            cls._messenger_registry[messenger_name] = subclass
            return subclass
        return decorator

    def __new__(cls, processor_name, *args, **kwargs):
        if processor_name not in cls._messenger_registry:
            raise ValueError(f"No messenger registered with name '{processor_name}'")
        return super(BaseProcessorMessenger, cls).__new__(cls._messenger_registry[processor_name])

    def __init__(self, role = None, content = None, *args, **kwargs):
        self.init_messager(role, content)

    def init_messenger(self, role: str = None, content: Union[str, Dict, List] = None):
        pass

    def update_message(self, role: str, content: Union[str, Dict, List]):
        pass

    def check_iter_round_num(self):
        pass

    def add_system_instruction(self, instruction: Union[str, Dict, List]):
        self.update_message("system", instruction)
    
    def add_assistant_message(self, message: Union[str, Dict, List]):
        self.update_message("assitant", message)

    def add_user_message(self, query: Union[str, Dict, List]):
        self.update_message("user", query)

    def add_user_image(self, image_base64: str):
        self.add_message("user", {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_base64}",
        })

    def add_feedback(self, feedback: Union[str, Dict, List]):
        self.add_message("system", feedback)

    def clear(self):
        self.messages.clear()

    def get_messages(self):
        return self.messages