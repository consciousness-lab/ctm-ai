import base64
import io
import os
from typing import Any, Union

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@BaseExecutor.register_executor('video_executor')
class VideoExecutor(BaseExecutor):
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

        video_paths = kwargs.get("video_frames")
        if not video_paths:
            raise ValueError(f"No video frames provided in kwargs, kwargs: {kwargs}")

        if not all(os.path.exists(path) for path in video_paths):
            missing_files = [path for path in video_paths if not os.path.exists(path)]
            raise FileNotFoundError(f"Some video frames not found: {missing_files}")

        base64_images = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"},
            }
            for path in video_paths
        ]

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
                            "text": f"{serialized_messages}\n\n[Attached Video Frames]",
                        },
                        *base64_images,
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
