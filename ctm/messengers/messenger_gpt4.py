from typing import Any, Dict, List, Optional, Tuple, Union

from .messenger_base import BaseMessenger


# Assuming BaseMessenger has a correctly typed decorator:
@BaseMessenger.register_messenger("gpt4_messenger")
class GPT4Messenger(BaseMessenger):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init_messenger(role, content)

    def init_messenger(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        # Define messages as a list of dictionaries with specific types
        self.messages: List[
            Dict[str, Union[str, Dict[str, Any], List[Any]]]
        ] = []
        if role is not None and content is not None:
            self.update_message(role, content)

    def update_message(
        self, role: str, content: Union[str, Dict[str, Any], List[Any]]
    ) -> None:
        # Append a new message to the list with a specified structure
        self.messages.append({"role": role, "content": content})

    def check_iter_round_num(self) -> int:
        # Return the number of iterations, which is the length of the messages list
        return len(self.messages)

    def get_messages(
        self,
    ) -> List[Dict[str, Union[str, Dict[str, Any], List[Any]]]]:
        return self.messages

    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any
    ):
        content = "Query: {}\n".format(query)
        if text is not None:
            content += "Text: {}\n".format(text)
        messages = [{"role": "user", "content": content}]
        return messages

    def update_executor_messages(self, gist: str):
        return
