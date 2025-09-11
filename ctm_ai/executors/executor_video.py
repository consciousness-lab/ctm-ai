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
        self.system_prompt = 'You are an expert in video analysis. Your task is to watch the provided video frames and answer questions about the events, objects, and actions depicted.'
        super().init_model(*args, **kwargs)

    def pil_to_base64(self, image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str

    def _ensure_user_tail(self, litellm_messages: List[dict]) -> int:
        for i in range(len(litellm_messages) - 1, -1, -1):
            if litellm_messages[i].get('role') == 'user':
                return i
        litellm_messages.append({'role': 'user', 'content': []})
        return len(litellm_messages) - 1

    def _ensure_content_list(self, msg: dict) -> None:
        c = msg.get('content')
        if c is None:
            msg['content'] = []
        elif isinstance(c, str):
            msg['content'] = [{'type': 'text', 'text': c}]
        elif isinstance(c, list):
            pass
        else:
            msg['content'] = [{'type': 'text', 'text': str(c)}]

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
        litellm_messages = [
            self.convert_message_to_litellm_format(msg) for msg in messages
        ]

        tail_idx = self._ensure_user_tail(litellm_messages)
        self._ensure_content_list(litellm_messages[tail_idx])

        # Add image frames to the message (convert PIL images to base64)
        for pil_image in pil_images:
            base64_image = self.pil_to_base64(pil_image)
            litellm_messages[tail_idx]['content'].append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                }
            )

        # Use the unified ask_base method
        return self.ask_base(
            messages=litellm_messages,
            model=model,
            default_additional_question='Would you like me to analyze any specific aspects of this video in more detail?',
        )
