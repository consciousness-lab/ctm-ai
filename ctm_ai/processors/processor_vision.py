import base64
import io
import os
from typing import Any, Dict, List

from ..utils import load_image
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('vision_processor')
class VisionProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        image_path = kwargs.get('image_path')
        if not image_path:
            return [{'role': 'assistant', 'content': ''}]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image file not found: {image_path}')
        base64_image = load_image(image_path)

        all_messages = [{'role': 'system', 'content': self.system_prompt}]

        image_message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': query},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                },
            ],
        }
        all_messages.append(image_message)

        return all_messages
