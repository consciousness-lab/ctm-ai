import json
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
        """Ask method for language processing using the base class functionality."""

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

        # Get response from base class
        response = self.ask_standard(
            messages=enhanced_messages,
            max_token=max_token,
            return_num=return_num,
            model=self.model_name,
        )

        # Parse JSON response
        try:
            parsed_response = json.loads(response.content)
            content = parsed_response.get('response', response.content)
            additional_question = parsed_response.get(
                'additional_question',
                'Would you like me to explain any specific aspects in more detail?',
            )
        except (json.JSONDecodeError, TypeError):
            # Fallback if JSON parsing fails
            content = response.content
            additional_question = (
                'Would you like me to explain any specific aspects in more detail?'
            )

        # Return response with additional question
        return Message(
            role=response.role,
            content=content,
            gist=content,
            gists=response.gists,
            additional_question=additional_question,
        )
