from typing import Any, Dict, List, Optional, Union

from .message import Message
from .messenger_base import BaseMessenger


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
        self.executor_messages: List[Message] = []
        self.scorer_messages: List[Message] = []

    def collect_executor_message(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        content = 'Query: {}\n'.format(query)
        if text is not None:
            content += 'Text: {}\n'.format(text)
        message = Message(
            role='user',
            content=content,
        )
        self.executor_messages.append(message)
        return message

    def parse_executor_message(self, *args: Any, **kwargs: Any) -> None:
        return

    def update_executor_message(self, gist: str) -> None:
        return
