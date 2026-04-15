import base64
import os
import subprocess
from typing import Any, Dict, List

from .processor_base import BaseProcessor

# Qwen VL API rejects videos shorter than ~4s with "video file is too short".
# For shorter clips we fall back to sending extracted frames as images.
_QWEN_MIN_VIDEO_DURATION_SEC = 4.0
_QWEN_FALLBACK_NUM_FRAMES = 4


def load_video_as_base64(video_path: str) -> str:
    """Load video file and convert to base64 string."""
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode('utf-8')


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds, or -1.0 on failure."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return -1.0


def _extract_frames_as_base64(
    video_path: str, num_frames: int = _QWEN_FALLBACK_NUM_FRAMES
) -> List[str]:
    """Extract evenly-spaced frames from video as base64 JPEG strings."""
    duration = _get_video_duration(video_path)
    if duration <= 0:
        duration = 1.0
    actual_frames = max(1, min(num_frames, int(duration / 0.1) or 1))

    frames_b64: List[str] = []
    for i in range(actual_frames):
        t = duration * (i + 0.5) / actual_frames
        t = max(0.0, min(t, max(0.0, duration - 0.01)))
        tmp = f'/tmp/ctm_qwen_frame_{os.getpid()}_{i}.jpg'
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-ss', str(t), '-i', video_path,
                 '-vframes', '1', '-q:v', '5', tmp],
                capture_output=True, timeout=10,
            )
            if os.path.exists(tmp):
                with open(tmp, 'rb') as f:
                    frames_b64.append(base64.b64encode(f.read()).decode('utf-8'))
                os.remove(tmp)
        except Exception:
            continue

    # Last-ditch fallback: grab the very first frame
    if not frames_b64:
        tmp = f'/tmp/ctm_qwen_frame_{os.getpid()}_fallback.jpg'
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-vframes', '1', '-q:v', '5', tmp],
                capture_output=True, timeout=10,
            )
            if os.path.exists(tmp):
                with open(tmp, 'rb') as f:
                    frames_b64.append(base64.b64encode(f.read()).decode('utf-8'))
                os.remove(tmp)
        except Exception:
            pass
    return frames_b64


@BaseProcessor.register_processor('video_processor')
class VideoProcessor(BaseProcessor):
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

        # Build video content block based on provider
        if self.provider == 'qwen':
            # Qwen VL needs videos to be long enough. For very short clips,
            # fall back to sending extracted frames as image_url blocks.
            duration = _get_video_duration(video_path)
            if 0 < duration < _QWEN_MIN_VIDEO_DURATION_SEC:
                frames_b64 = _extract_frames_as_base64(video_path)
                media_content_blocks = [
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{fb64}'},
                    }
                    for fb64 in frames_b64
                ]
                if not media_content_blocks:
                    # Frame extraction failed – fall back to sending the video.
                    base64_video = load_video_as_base64(video_path)
                    media_content_blocks = [{
                        'type': 'video_url',
                        'video_url': {
                            'url': f'data:{mime_type};base64,{base64_video}'
                        },
                    }]
            else:
                base64_video = load_video_as_base64(video_path)
                media_content_blocks = [{
                    'type': 'video_url',
                    'video_url': {
                        'url': f'data:{mime_type};base64,{base64_video}'
                    },
                }]
        else:
            # Gemini via litellm uses image_url type for video
            base64_video = load_video_as_base64(video_path)
            media_content_blocks = [{
                'type': 'image_url',
                'image_url': {
                    'url': f'data:{mime_type};base64,{base64_video}'
                },
            }]

        video_message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': f'{query}\n'},
                *media_content_blocks,
            ],
        }

        all_messages = [
            {'role': 'system', 'content': self.system_prompt},
            video_message,
        ]

        return all_messages
