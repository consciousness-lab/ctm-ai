import os
from typing import Any, List

from litellm import completion

from ..messengers import Message
from ..utils import load_images, message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('video_executor')
class VideoExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using LiteLLM for Gemini."""
        # Set default model to Gemini for video processing
        kwargs.setdefault('model', 'gemini/gemini-1.5-flash-8b')
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
        """Ask method for video processing using LiteLLM with Gemini."""
        model = model or self.model_name

        video_frames_path = kwargs.get('video_frames_path')
        if not video_frames_path:
            return Message(role='assistant', content='', gist='', gists=[])

        if not all(os.path.exists(path) for path in video_frames_path):
            missing_files = [
                path for path in video_frames_path if not os.path.exists(path)
            ]
            raise FileNotFoundError(f'Some video frames not found: {missing_files}')

        # Load video frames
        images = load_images(video_frames_path)

        # Convert messages to text
        message_text = ' '.join([msg.content for msg in messages if msg.content])

        # Create prompt for video analysis
        prompt = f'{message_text}. Here are the relevant image frames of the video:'

        # Create message with video frames for Gemini
        video_message = {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}

        # Add image frames to the message
        for image in images:
            video_message['content'].append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{image}'},
                }
            )

        # Use LiteLLM completion with Gemini for video
        response = completion(
            model=model,
            messages=[video_message],
            max_tokens=max_token,
            n=return_num,
        )

        # Extract responses from all candidates
        gists = [
            response.choices[i].message.content for i in range(len(response.choices))
        ]

        return Message(
            role='assistant',
            content=gists[0],
            gist=gists[0],
            gists=gists,
        )
