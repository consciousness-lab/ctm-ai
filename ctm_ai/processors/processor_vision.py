import base64
import io
import os
from typing import Any, Dict, List

from ..utils import load_image
from .processor_base import BaseProcessor


def pil_to_base64(image) -> str:
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


@BaseProcessor.register_processor('vision_processor')
class VisionProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        image_path = kwargs.get('image_path')
        image = kwargs.get('image')
        if not image_path and not image:
            return None
        if image_path:
            base64_image = load_image(image_path)
        if image:
            base64_image = pil_to_base64(image)

        # Use system_prompt from config if provided, otherwise use default
        if not self.system_prompt:
            self.system_prompt = 'You are an expert in image understanding. Your task is to analyze the provided image and answer questions about it.'

        all_messages = [{'role': 'system', 'content': self.system_prompt}]

        image_message = {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f'{query}\n',
                },
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                },
            ],
        }
        all_messages.append(image_message)

        return all_messages
