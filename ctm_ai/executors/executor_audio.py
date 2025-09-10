import base64
import os
from typing import Any, List

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('audio_executor')
class AudioExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model for audio processing."""
        # Set default system prompt for audio analysis
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

    def _ensure_user_tail(self, litellm_messages: List[dict]) -> int:
        for i in range(len(litellm_messages) - 1, -1, -1):
            if litellm_messages[i].get('role') == 'user':
                return i
        litellm_messages.append({'role': 'user', 'content': []})
        return len(litellm_messages) - 1

    def _ensure_content_list(self, msg: dict) -> None:
        c = msg.get('content')
        if c is None:
            msg['content'] = []
        elif isinstance(c, str):
            msg['content'] = [{'type': 'text', 'text': c}]
        elif isinstance(c, list):
            pass
        else:
            msg['content'] = [{'type': 'text', 'text': str(c)}]

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

        litellm_messages = [
            self.convert_message_to_litellm_format(msg) for msg in messages
        ]

        tail_idx = self._ensure_user_tail(litellm_messages)
        self._ensure_content_list(litellm_messages[tail_idx])

        try:
            # Get MIME type for audio file
            mime_type = self.get_mime_type(audio_path)

            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            encoded_data = base64.b64encode(audio_bytes).decode('utf-8')

            litellm_messages[tail_idx]['content'].append(
                {
                    'type': 'file',
                    'file': {
                        'file_data': f'data:{mime_type};base64,{encoded_data}',
                    },
                }
            )

            # Use the unified ask_base method
            return self.ask_base(
                messages=litellm_messages,
                model=model,
                default_additional_question='Would you like me to analyze any specific aspects of this audio in more detail?',
            )

        except Exception as e:
            raise RuntimeError(f'Error processing audio: {str(e)}')
