import os
from typing import Any, List

from litellm import completion

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('audio_executor')
class AudioExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using LiteLLM for Gemini."""
        # Set default model to Gemini for audio processing
        kwargs.setdefault('model', 'gemini/gemini-1.5-flash')
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
        max_token: int = 300,
        return_num: int = 1,
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for audio processing using LiteLLM with Gemini."""
        model = model or self.model_name

        if not messages:
            raise ValueError('No messages provided')

        audio_path = kwargs.get('audio_path')
        if not audio_path:
            return Message(role='assistant', content='', gist='', gists=[])

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'Audio file not found: {audio_path}')

        query = messages[-1].content

        try:
            # For audio processing with Gemini through LiteLLM
            # We need to prepare the message with audio data
            mime_type = self.get_mime_type(audio_path)

            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            # Create message with audio content for Gemini
            audio_message = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': query},
                    {
                        'type': 'audio_url',
                        'audio_url': {
                            'url': f'data:{mime_type};base64,{audio_data.hex()}'
                        },
                    },
                ],
            }

            # Use LiteLLM completion with Gemini for audio
            response = completion(
                model=model,
                messages=[audio_message],
                max_tokens=max_token,
                n=return_num,
            )

            content = response.choices[0].message.content

            return Message(
                role='assistant',
                content=content,
                gist=content,
                gists=[content],
            )

        except Exception as e:
            raise RuntimeError(f'Error processing audio: {str(e)}')
