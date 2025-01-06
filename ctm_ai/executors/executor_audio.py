from typing import Any, List
import os
import google.generativeai as genai

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('audio_executor')
class AudioExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')
        self.supported_formats = {'mp3', 'wav', 'aac', 'flac', 'mp4'}
        self.mime_types = {
            'mp3': 'audio/mp3',
            'wav': 'audio/wav',
            'aac': 'audio/aac',
            'flac': 'audio/flac',
            'mp4': 'audio/mp4',
        }

    def get_mime_type(self, file_path: str) -> str:
        extension = file_path.split('.')[-1].lower()
        if extension not in self.mime_types:
            raise ValueError(f"Unsupported audio format: {extension}")
        return self.mime_types[extension]

    @message_exponential_backoff()
    def ask(
            self,
            messages: List[Message],
            *args: Any,
            **kwargs: Any
    ) -> Message:
        if not messages:
            raise ValueError("No messages provided")

        audio_path = kwargs.get('audio_path')
        if not audio_path:
            raise ValueError("No audio path provided in kwargs")

        query = messages[-1].content

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            mime_type = self.get_mime_type(audio_path)

            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            response = self.model.generate_content([
                query,
                {
                    "mime_type": mime_type,
                    "data": audio_data
                }
            ])

            return Message(
                role='assistant',
                content=response.text,
                gist=response.text,
                gists=[response.text],
            )

        except Exception as e:
            raise RuntimeError(f"Error processing audio: {str(e)}")
