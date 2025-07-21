import os
from typing import Any, List

from ctm_ai.utils.loader import load_image

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('vision_executor')
class VisionExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using the base class functionality."""
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
        """Ask method for vision processing with image handling using unified ask_base."""
        model = model or self.model_name

        # Handle image path
        image_path = kwargs.get('image_path')
        if not image_path:
            return Message(
                role='assistant',
                content='',
                gist='',
                additional_question='Please provide an image to analyze.',
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image file not found: {image_path}')

        # Load and add image to messages
        base64_image = load_image(image_path)

        # Create enhanced prompt for JSON response
        original_query = messages[-1].content if messages else ''
        enhanced_prompt = f"""{original_query}

Please respond in JSON format with the following structure:
{{
    "response": "Your detailed analysis of the image",
    "additional_question": "A follow-up question to gather more specific information about what the user wants to know about the image"
}}

Your additional_question should be specific to image analysis, such as asking about particular objects, areas, colors, relationships, or details in the image."""

        # Create message with image for LiteLLM
        image_message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': enhanced_prompt},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                },
            ],
        }

        # Use the unified ask_base method
        return self.ask_base(
            messages=[image_message],
            max_token=max_token,
            return_num=return_num,
            model=model,
            default_additional_question='Would you like me to analyze any specific aspects of this image in more detail?',
        )
