import base64
import os
from typing import Any, List

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('audio_executor')
class AudioExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using LiteLLM for Gemini."""
        # Set default model to Gemini for audio processing
        kwargs.setdefault('model', 'gemini/gemini-1.5-flash')
        self.system_prompt = 'You are an expert in audio analysis. Your task is to listen to the provided audio and answer questions about its content, such as tone, emotion, or spoken words.'
        super().init_model(*args, **kwargs)

        # Configure supported formats
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

    @message_exponential_backoff()
    def ask(
        self,
        messages: List[Message],
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for audio processing using unified ask_base."""
        model = model or self.model_name

        if not messages:
            raise ValueError('No messages provided')

        audio_path = kwargs.get('audio_path')
        if not audio_path:
            return Message(
                role='assistant',
                gist='',
                additional_question='Please provide an audio file to analyze.',
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'Audio file not found: {audio_path}')

        query = messages[-1].content

        try:
            # Get MIME type for audio file
            mime_type = self.get_mime_type(audio_path)

            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            encoded_data = base64.b64encode(audio_bytes).decode('utf-8')

            # Create message with audio content for LiteLLM (following Gemini docs)
            audio_message = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': query},
                    {
                        'type': 'file',
                        'file': {
                            'file_data': f'data:{mime_type};base64,{encoded_data}',
                        },
                    },
                ],
            }

            # Use the unified ask_base method
            return self.ask_base(
                messages=[audio_message],
                model=model,
                default_additional_question='Would you like me to analyze any specific aspects of this audio in more detail?',
            )

        except Exception as e:
            raise RuntimeError(f'Error processing audio: {str(e)}')
