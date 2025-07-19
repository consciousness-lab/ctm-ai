import os
from typing import Any, List

from litellm import completion

from ctm_ai.utils.loader import load_image

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('vision_executor')
class VisionExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using the base class functionality."""
        super().init_model(*args, **kwargs)

    @message_exponential_backoff()
    def ask(
        self,
        messages: List[Message],
        max_token: int = 300,
        return_num: int = 5,
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for vision processing with image handling using LiteLLM."""
        model = model or self.model_name

        # Handle image path
        image_path = kwargs.get('image_path')
        if not image_path:
            return Message(role='assistant', content='', gist='', gists=[])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image file not found: {image_path}')

        # Convert messages to LiteLLM format
        litellm_messages = [
            self.convert_message_to_litellm_format(message) for message in messages
        ]

        # Load and add image to messages
        base64_image = load_image(image_path)
        image_message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Here is the relevant image:'},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                },
            ],
        }
        litellm_messages.append(image_message)  # type: ignore[arg-type]

        # Use LiteLLM completion with vision support
        response = completion(
            model=model,
            messages=litellm_messages,
            max_tokens=max_token,
            n=return_num,
        )

        gists = [response.choices[i].message.content for i in range(return_num)]

        return Message(
            role='assistant',
            content=gists[0],
            gist=gists[0],
            gists=gists,
        )
