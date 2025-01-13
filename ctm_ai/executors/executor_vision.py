import base64
import os
from typing import Any, Union

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .executor_base import BaseExecutor
from ..messengers import Message
from ..utils import message_exponential_backoff


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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

        image_path = kwargs.get("image")
        if not image_path:
            raise ValueError(f"No image path provided in kwargs, kwargs: {kwargs}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        base64_image = encode_image(image_path)

        serialized_messages = "\n".join(
            [
                f"[{msg.role.upper()}]: {msg.content}"
                for msg in model_messages
                if hasattr(msg, "role") and hasattr(msg, "content")
            ]
        )

        response = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{serialized_messages}\n\n[Attached Image]",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=max_token,
            n=return_num,
        )

        gists = [response.choices[i].message.content for i in range(return_num)]
        return Message(
            role="assistant",
            content=gists[0],
            gist=gists[0],
            gists=gists,
        )
