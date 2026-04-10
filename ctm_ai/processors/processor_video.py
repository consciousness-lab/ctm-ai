import base64
import os
import tempfile
from typing import Any, Dict, List

from .processor_base import BaseProcessor


def load_video_as_base64(video_path: str) -> str:
    """Load video file and convert to base64 string."""
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode('utf-8')


@BaseProcessor.register_processor('video_processor')
class VideoProcessor(BaseProcessor):
    def _build_frames_messages(
        self, query: str, video_path: str
    ) -> List[Dict[str, Any]]:
        """Extract frames from video and build messages with images instead of video.

        Used as fallback when the video is too long/short for the model's API.
        """
        from ..utils.loader import extract_video_frames

        tmp_dir = tempfile.mkdtemp(prefix='ctm_frames_')
        try:
            frame_paths = extract_video_frames(
                video_path, tmp_dir, max_frames=self.max_frames
            )
        except Exception:
            frame_paths = []

        if not frame_paths:
            return None

        content_parts = [
            {
                'type': 'text',
                'text': (
                    f'{query}\n\n'
                    f'Below are {len(frame_paths)} frames extracted from the video. '
                    f'Analyze them as if watching the video.'
                ),
            }
        ]

        for fp in frame_paths:
            with open(fp, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            content_parts.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{encoded}'},
                }
            )

        # Clean up temp files
        for fp in frame_paths:
            try:
                os.unlink(fp)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': content_parts},
        ]

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        # Use system_prompt from config if provided, otherwise use default
        if not self.system_prompt:
            self.system_prompt = 'You are an expert in video understanding. Your task is to analyze the provided video and answer questions about it.'

        video_path = kwargs.get('video_path')
        if not video_path:
            return None

        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Video file not found: {video_path}')

        # If use_frames is enabled, extract frames and send as images
        if self.use_frames:
            return self._build_frames_messages(query, video_path)

        # Check file size (inline data limit is 20MB)
        file_size = os.path.getsize(video_path)
        max_size = 20 * 1024 * 1024  # 20MB in bytes
        if file_size > max_size:
            raise ValueError(
                f'Video file size ({file_size / 1024 / 1024:.2f}MB) exceeds '
                f'the 20MB limit for inline video data. '
                f'Please use a smaller video file.'
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

        # Build video content block based on provider
        data_url = f'data:{mime_type};base64,{base64_video}'
        if self.provider == 'qwen':
            # Qwen uses OpenAI-compatible video_url type
            video_content = {
                'type': 'video_url',
                'video_url': {'url': data_url},
            }
        else:
            # Gemini via litellm uses image_url type for video
            video_content = {
                'type': 'image_url',
                'image_url': {'url': data_url},
            }

        video_message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': f'{query}\n'},
                video_content,
            ],
        }

        all_messages = [
            {'role': 'system', 'content': self.system_prompt},
            video_message,
        ]

        return all_messages
