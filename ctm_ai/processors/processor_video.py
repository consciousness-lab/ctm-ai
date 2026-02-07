import base64
import os
from typing import Any, Dict, List

from .processor_base import BaseProcessor


def load_video_as_base64(video_path: str) -> str:
    """Load video file and convert to base64 string."""
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode('utf-8')


@BaseProcessor.register_processor("video_processor")
class VideoProcessor(BaseProcessor):
    REQUIRED_KEYS = ["GEMINI_API_KEY"]

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.system_prompt = "You are an expert in video understanding. Your task is to analyze the provided video and answer questions about it."
        
        video_path = kwargs.get("video_path")
        if not video_path:
            return None
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file size (Gemini inline data limit is 20MB)
        file_size = os.path.getsize(video_path)
        max_size = 20 * 1024 * 1024  # 20MB in bytes
        if file_size > max_size:
            raise ValueError(
                f"Video file size ({file_size / 1024 / 1024:.2f}MB) exceeds "
                f"the 20MB limit for inline video data. "
                f"Please use a smaller video file."
            )
        
        # Detect MIME type from file extension
        ext = os.path.splitext(video_path)[1].lower()
        mime_type_map = {
            '.mp4': 'video/mp4',
            '.mpeg': 'video/mpeg',
            '.mov': 'video/mov',
            '.avi': 'video/avi',
            '.flv': 'video/x-flv',
            '.mpg': 'video/mpg',
            '.webm': 'video/webm',
            '.wmv': 'video/wmv',
            '.3gp': 'video/3gpp',
        }
        mime_type = mime_type_map.get(ext, 'video/mp4')
        
        # Load and encode video
        base64_video = load_video_as_base64(video_path)
        
        # Build message with inline video data
        video_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{query}\n",
                },
                {
                    "type": "image_url",  # litellm uses image_url type for video as well
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_video}"
                    }
                }
            ],
        }
        
        all_messages = [
            {"role": "system", "content": self.system_prompt},
            video_message
        ]
        
        return all_messages
