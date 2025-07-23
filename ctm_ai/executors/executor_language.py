from typing import Any, List

from ..messengers import Message
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('language_executor')
class LanguageExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using the base class functionality."""
        self.system_prompt = 'You are an expert in language understanding. Your task is to analyze the provided text and answer questions about it.'
        super().init_model(*args, **kwargs)
        self.model_name = kwargs.get("language_model", "gemini/gemini-2.0-flash-lite")

    def ask(
        self,
        messages: List[Message],
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Ask method for language processing using the unified ask_base method."""

        # Convert messages to LiteLLM format
        litellm_messages = [
            self.convert_message_to_litellm_format(msg) for msg in messages
        ]

        # Use the unified ask_base method
        return self.ask_base(
            messages=litellm_messages,
            model=self.model_name,
            default_additional_question='Would you like me to explain any specific aspects in more detail?',
        )
