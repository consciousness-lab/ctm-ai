import os
from typing import Any, List

from ..messengers import Message
from ..utils import load_images, message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('video_executor')
class VideoExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using LiteLLM for Gemini."""
        # Set default model to Gemini for video processing
        kwargs.setdefault('model', 'gemini/gemini-1.5-flash-8b')
        super().init_model(*args, **kwargs)

    @message_exponential_backoff()
    def ask(
        self,
        messages: List[Message],
        max_token: int = 300,
        return_num: int = 5,
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for video processing using unified ask_base."""
        model = model or self.model_name

        video_frames_path = kwargs.get('video_frames_path')
        if not video_frames_path:
            return Message(
                role='assistant',
                content='',
                gist='',
                additional_question='Please provide video frames to analyze.',
            )

        if not all(os.path.exists(path) for path in video_frames_path):
            missing_files = [
                path for path in video_frames_path if not os.path.exists(path)
            ]
            raise FileNotFoundError(f'Some video frames not found: {missing_files}')

        # Load video frames
        images = load_images(video_frames_path)

        # Convert messages to text
        message_text = ' '.join([msg.content for msg in messages if msg.content])

        # Create enhanced prompt for JSON response
        enhanced_prompt = f"""{message_text}

Please respond in JSON format with the following structure:
{{
    "response": "Your detailed analysis of the video frames",
    "additional_question": "A follow-up question to gather more specific information about what the user wants to know about the video"
}}

Your additional_question should be specific to video analysis, such as asking about specific frames, time periods, actions, movements, scene transitions, or objects throughout the video sequence. Here are the relevant image frames of the video:"""

        # Create message with video frames for LiteLLM
        video_message = {
            'role': 'user',
            'content': [{'type': 'text', 'text': enhanced_prompt}],
        }

        # Add image frames to the message
        for image in images:
            video_message['content'].append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{image}'},
                }
            )

        # Use the unified ask_base method
        return self.ask_base(
            messages=[video_message],
            max_token=max_token,
            return_num=return_num,
            model=model,
            default_additional_question='Would you like me to analyze any specific aspects of this video in more detail?',
        )
