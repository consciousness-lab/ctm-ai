import base64
import os
from typing import Any, Dict, List

from .processor_base import BaseProcessor


@BaseProcessor.register_processor('audio_processor')
class AudioProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

    def _init_info(self, *args: Any, **kwargs: Any) -> None:
        self.system_prompt = 'You are an expert in audio analysis. Your task is to listen to the provided audio and answer questions about its content, such as tone, emotion, or spoken words.'
        self.supported_formats = {'mp3', 'wav', 'aac', 'flac', 'mp4'}
        self.mime_types = {
            'mp3': 'audio/mp3',
            'wav': 'audio/wav',
            'aac': 'audio/aac',
            'flac': 'audio/flac',
            'mp4': 'audio/mp4',
        }

    def get_mime_type(self, file_path: str) -> str:
        """Get MIME type for audio file."""
        extension = file_path.split('.')[-1].lower()
        if extension not in self.mime_types:
            raise ValueError(f'Unsupported audio format: {extension}')
        return self.mime_types[extension]

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        audio_path = kwargs.get('audio_path')
        print(f"[DEBUG AudioProcessor] audio_path received: {audio_path}")
        if not audio_path:
            # 没有音频输入时返回 None，让上层跳过这个处理器
            print("[DEBUG AudioProcessor] No audio_path, returning None")
            return None
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'Audio file not found: {audio_path}')
        try:
            mime_type = self.get_mime_type(audio_path)
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            encoded_data = base64.b64encode(audio_bytes).decode('utf-8')
            audio_message = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': query},
                    {
                        'type': 'file',
                        'file': {
                            'file_data': f'data:{mime_type};base64,{encoded_data}'
                        },
                    },
                ],
            }

            all_messages = [{'role': 'system', 'content': self.system_prompt}]
            all_messages.append(audio_message)
            return all_messages
        except Exception as e:
            raise RuntimeError(f'Error building executor messages: {str(e)}')
