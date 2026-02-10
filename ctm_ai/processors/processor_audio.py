import base64
import os
import subprocess
import tempfile
from typing import Any, Dict, List

from .processor_base import BaseProcessor


@BaseProcessor.register_processor('audio_processor')
class AudioProcessor(BaseProcessor):
    REQUIRED_KEYS = []

    def _init_info(self, *args: Any, **kwargs: Any) -> None:
        # Use system_prompt from config if provided, otherwise use default
        if not self.system_prompt:
            self.system_prompt = 'You are an expert in audio analysis. You have been provided with an audio file. Listen to the audio carefully and analyze its tone, emotion, pitch, speed, vocal patterns, and any sarcastic cues. Answer the query based on what you hear in the audio. Do NOT say you need audio analysis - you already have the audio.'
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

    @staticmethod
    def _make_black_video_with_audio(audio_path: str, output_path: str) -> bool:
        """Convert audio file to a video with a black screen using ffmpeg.

        Qwen-Omni requires audio to be embedded in a video for processing.
        """
        cmd = [
            'ffmpeg',
            '-y',
            '-f',
            'lavfi',
            '-i',
            'color=c=black:s=320x240:r=1',
            '-i',
            audio_path,
            '-shortest',
            '-c:v',
            'libx264',
            '-tune',
            'stillimage',
            '-c:a',
            'aac',
            '-pix_fmt',
            'yuv420p',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        return True

    def _build_gemini_audio_content(
        self, audio_path: str, query: str
    ) -> Dict[str, Any]:
        """Build audio message in Gemini format (file type - litellm convention)."""
        mime_type = self.get_mime_type(audio_path)
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        encoded_data = base64.b64encode(audio_bytes).decode('utf-8')
        return {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f'[AUDIO PROVIDED BELOW]\n\n{query}\n\nBased on the audio you received, provide your analysis.',
                },
                {
                    'type': 'file',
                    'file': {
                        'file_data': f'data:{mime_type};base64,{encoded_data}',
                    },
                },
            ],
        }

    def _build_qwen_audio_content(self, audio_path: str, query: str) -> Dict[str, Any]:
        """Build audio message in Qwen format (audio embedded in black-screen video)."""
        tmp_video = tempfile.mktemp(suffix='.mp4')
        try:
            if not self._make_black_video_with_audio(audio_path, tmp_video):
                raise RuntimeError(
                    'Failed to convert audio to video with ffmpeg. '
                    'Ensure ffmpeg is installed.'
                )
            with open(tmp_video, 'rb') as f:
                video_bytes = f.read()
            encoded_data = base64.b64encode(video_bytes).decode('utf-8')
        finally:
            if os.path.exists(tmp_video):
                os.unlink(tmp_video)

        return {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': (
                        f'Focus ONLY on what the person is SAYING in the audio. '
                        f'The video is just a black screen.\n\n{query}\n\n'
                        f'Based on the audio you received, provide your analysis.'
                    ),
                },
                {
                    'type': 'video_url',
                    'video_url': {'url': f'data:video/mp4;base64,{encoded_data}'},
                },
            ],
        }

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        audio_path = kwargs.get('audio_path')
        if not audio_path:
            return None
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'Audio file not found: {audio_path}')
        try:
            if self.provider == 'qwen':
                audio_message = self._build_qwen_audio_content(audio_path, query)
            else:
                audio_message = self._build_gemini_audio_content(audio_path, query)

            all_messages = [{'role': 'system', 'content': self.system_prompt}]
            all_messages.append(audio_message)
            return all_messages
        except Exception as e:
            raise RuntimeError(f'Error building executor messages: {str(e)}')
