import base64
import io
import os
from typing import Any, Dict, List

from ..utils import load_images
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('video_processor')
class VideoProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

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
        video_frames_path = kwargs.get('video_frames_path')
        if not video_frames_path:
            return None
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
        all_messages = [video_message]

        all_messages.insert(0, {'role': 'system', 'content': self.system_prompt})

        return all_messages
