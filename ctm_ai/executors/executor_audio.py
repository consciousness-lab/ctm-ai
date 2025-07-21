import json
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
            return Message(
                role='assistant',
                content='',
                gist='',
                additional_question='Please provide an audio file to analyze.',
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'Audio file not found: {audio_path}')

        query = messages[-1].content

        # Create enhanced prompt for JSON response
        enhanced_query = f"""{query}

Please respond in JSON format with the following structure:
{{
    "response": "Your detailed analysis of the audio",
    "additional_question": "A follow-up question to gather more specific information about what the user wants to know about the audio"
}}

Your additional_question should be specific to audio analysis, such as asking about time segments, specific sounds, audio quality, voices, or emotional content."""

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
                    {'type': 'text', 'text': enhanced_query},
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

            # Parse JSON response
            try:
                parsed_response = json.loads(content)
                response_text = parsed_response.get('response', content)
                additional_question = parsed_response.get(
                    'additional_question',
                    'Would you like me to analyze any specific aspects of this audio in more detail?',
                )
            except (json.JSONDecodeError, TypeError):
                # Fallback if JSON parsing fails
                response_text = content
                additional_question = 'Would you like me to analyze any specific aspects of this audio in more detail?'

            return Message(
                role='assistant',
                content=response_text,
                gist=response_text,
                gists=[response_text],
                additional_question=additional_question,
            )

        except Exception as e:
            raise RuntimeError(f'Error processing audio: {str(e)}')
