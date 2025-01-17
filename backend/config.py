import os
from typing import Dict, Set


class Config:
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER: str = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH: int = 1 * 1024 * 1024 * 1024  # 1GB
    ALLOWED_EXTENSIONS: Dict[str, Set[str]] = {
        'images': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
        'audios': {'mp3', 'wav', 'aac', 'flac', 'mp4'},
        'videos': {'mp4', 'avi', 'mov', 'wmv', 'flv'},
    }
    FRONTEND_TO_BACKEND_PROCESSORS: Dict[str, str] = {
        'VisionProcessor': 'vision_processor',
        'LanguageProcessor': 'language_processor',
        'SearchProcessor': 'search_processor',
        'MathProcessor': 'math_processor',
        'CodeProcessor': 'code_processor',
        'AudioProcessor': 'audio_processor',
        'VideoProcessor': 'video_processor',
    }
