import base64
import io
import os
from typing import Any, Dict, List

from ..utils import load_images
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('video_processor')
class VideoProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

    def _init_info(self, *args: Any, **kwargs: Any) -> None:
        self.system_prompt = 'You are an expert in video analysis. Your task is to watch the provided video frames and answer questions about the events, objects, and actions depicted.'

    def pil_to_base64(self, image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        video_frames_path = kwargs.get('video_frames_path')
        if not video_frames_path:
            return [{'role': 'assistant', 'content': ''}]
        if not all(os.path.exists(path) for path in video_frames_path):
            missing_files = [
                path for path in video_frames_path if not os.path.exists(path)
            ]
            raise FileNotFoundError(f'Some video frames not found: {missing_files}')
        pil_images = load_images(video_frames_path)
        video_message = {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f'Query: {query}\nNote: The input contains {len(video_frames_path)} video frames. Please integrate visual information across these frames for a comprehensive analysis.\n',
                }
            ],
        }

        for pil_image in pil_images:
            base64_image = self.pil_to_base64(pil_image)
            video_message['content'].append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                }
            )

        return [video_message]
