import os
from typing import Any, List

import google.generativeai as genai
from PIL import Image

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


def load_images(image_paths: List[str]) -> List[Image.Image]:
    if not image_paths:
        raise ValueError('No images provided')

    image_paths_sorted = sorted(image_paths)

    images = []
    for img_path in image_paths_sorted:
        try:
            image = Image.open(img_path)
            images.append(image)
        except Exception as e:
            raise RuntimeError(f'Failed to load image {img_path}: {e}')
    return images


@BaseExecutor.register_executor('video_executor')
class VideoExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')

    def convert_message_to_param(self, message: Message) -> str:
        if message.content is None:
            raise ValueError('Message content cannot be None')
        return message.content

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

        video_frames_path = kwargs.get('video_frames_path')
        if not video_frames_path:
            return Message(role='assistant', content='', gist='', gists=[])

        images = load_images(video_frames_path)

        prompt = f'{model_messages}. Here are the relevant image frames of the video:'

        inputs = [prompt] + images

        gen_config = genai.types.GenerationConfig(
            candidate_count=return_num, max_output_tokens=max_token
        )
        response = self.model.generate_content(
            inputs,
            generation_config=gen_config,
        )

        gists = [candidate.content.parts[0].text for candidate in response.candidates]
        return Message(
            role='assistant',
            content=gists[0],
            gist=gists[0],
            gists=gists,
        )
