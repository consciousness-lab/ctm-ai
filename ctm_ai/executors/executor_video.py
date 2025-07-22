import base64
import io
import os
from typing import Any, List

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

    def pil_to_base64(self, image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str

    @message_exponential_backoff()
    def ask(
        self,
        messages: List[Message],
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for video processing using unified ask_base."""
        model = model or self.model_name

        video_frames_path = kwargs.get('video_frames_path')
        if not video_frames_path:
            return Message(
                role='assistant',
                content='',
                gist='',
                additional_question='Please provide video frames to analyze.',
            )

        if not all(os.path.exists(path) for path in video_frames_path):
            missing_files = [
                path for path in video_frames_path if not os.path.exists(path)
            ]
            raise FileNotFoundError(f'Some video frames not found: {missing_files}')

        # Load video frames (returns PIL Image objects)
        pil_images = load_images(video_frames_path)

        # Convert messages to text
        message_text = ' '.join([msg.content for msg in messages if msg.content])

        # Create message with video frames for LiteLLM (following Gemini docs)
        video_message = {
            'role': 'user',
            'content': [{'type': 'text', 'text': message_text}],
        }

        # Add image frames to the message (convert PIL images to base64)
        for pil_image in pil_images:
            base64_image = self.pil_to_base64(pil_image)
            video_message['content'].append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                }
            )

        # Use the unified ask_base method
        return self.ask_base(
            messages=[video_message],
            model=model,
            default_additional_question='Would you like me to analyze any specific aspects of this video in more detail?',
        )
