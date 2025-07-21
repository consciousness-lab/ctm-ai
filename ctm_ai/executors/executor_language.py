from typing import Any, List

from ..messengers import Message
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('language_executor')
class LanguageExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using the base class functionality."""
        super().init_model(*args, **kwargs)

    def ask(
        self,
        messages: List[Message],
        max_token: int = 300,
        return_num: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for language processing using the unified ask_base method."""

        # Enhance the last message to request JSON format
        if messages:
            enhanced_messages = messages[:-1].copy()
            last_message = messages[-1]

            enhanced_content = f"""{last_message.content}

Please respond in JSON format with the following structure:
{{
    "response": "Your detailed response to the query",
    "additional_question": "A follow-up question to gather more specific information or explore related topics"
}}

Your additional_question should help clarify what specific aspects the user wants to know more about or suggest related topics that might be of interest."""

            enhanced_last_message = Message(
                role=last_message.role, content=enhanced_content
            )
            enhanced_messages.append(enhanced_last_message)
        else:
            enhanced_messages = messages

        # Convert messages to LiteLLM format
        litellm_messages = [
            self.convert_message_to_litellm_format(msg) for msg in enhanced_messages
        ]

        # Use the unified ask_base method
        return self.ask_base(
            messages=litellm_messages,
            max_token=max_token,
            return_num=return_num,
            model=self.model_name,
            default_additional_question='Would you like me to explain any specific aspects in more detail?',
        )
