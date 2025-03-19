import os
from typing import Any, Union

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ctm_ai.utils.loader import load_image

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('vision_executor')
class VisionExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.model = OpenAI()

    def convert_message_to_param(
        self, message: Message
    ) -> Union[
        ChatCompletionAssistantMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
    ]:
        if message.content is None:
            raise ValueError('Message content cannot be None')
        if message.role == 'system':
            return ChatCompletionSystemMessageParam(
                role='system', content=message.content
            )
        elif message.role == 'user':
            return ChatCompletionUserMessageParam(role='user', content=message.content)
        elif message.role == 'assistant':
            return ChatCompletionAssistantMessageParam(
                role='assistant', content=message.content
            )
        else:
            raise ValueError(f'Unsupported message role: {message.role}')

    @message_exponential_backoff()
    def ask(
        self,
        messages: list[Message],
        max_token: int = 300,
        return_num: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        model_messages = [
            self.convert_message_to_param(message) for message in messages
        ]

        image_path = kwargs.get('image_path')
        if not image_path:
            return Message(role='assistant', content='', gist='', gists=[])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image file not found: {image_path}')

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
        model_messages.append(image_message)  # type: ignore[arg-type]
        response = self.model.chat.completions.create(
            model='gpt-4o',
            messages=model_messages,
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
