from typing import Any, Dict, List, Optional, Union

from .messenger_base import BaseMessenger


# Assuming BaseMessenger has a correctly typed decorator:
@BaseMessenger.register_messenger('gpt4_messenger')
class GPT4Messenger(BaseMessenger):
    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init_messenger(role, content)

    def init_messenger(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.messages: List[Dict[str, Union[str, Dict[str, Any], List[Any]]]] = []

    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        content = 'Query: {}\n'.format(query)
        if text is not None:
            content += 'Text: {}\n'.format(text)
        messages = [{'role': 'user', 'content': content}]
        return messages

    def update_executor_messages(self, gist: str) -> None:
        return
