import json
from typing import Any, Callable, Dict, List, Type

from litellm import completion

from ..messengers import Message
from ..utils import (
    call_llm,
    configure_litellm,
    convert_message_to_litellm_format,
    message_exponential_backoff,
)


class BaseExecutor(object):
    _executor_registry: Dict[str, Type['BaseExecutor']] = {}

    @classmethod
    def register_executor(
        cls, name: str
    ) -> Callable[[Type['BaseExecutor']], Type['BaseExecutor']]:
        def decorator(
            subclass: Type['BaseExecutor'],
        ) -> Type['BaseExecutor']:
            cls._executor_registry[name] = subclass
            return subclass

        return decorator

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> 'BaseExecutor':
        if name not in cls._executor_registry:
            raise ValueError(f"No executor registered with name '{name}'")
        instance = super(BaseExecutor, cls).__new__(cls._executor_registry[name])
        instance.name = name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_model(*args, **kwargs)

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model. Can be overridden by subclasses."""
        # Default configuration for LiteLLM
        self.model_name = kwargs.get('model', 'gpt-4o')
        self.try_times = kwargs.get('try_times', 3)
        self.default_max_tokens = kwargs.get('max_tokens', 4096)
        self.default_return_num = kwargs.get('return_num', 1)
        self.default_temperature = kwargs.get('temperature', 0.0)

        # Get Gemini API key if provided
        self.gemini_api_key = kwargs.get('gemini_api_key')

        # Configure LiteLLM settings
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Configure LiteLLM settings and callbacks."""
        configure_litellm(
            model_name=self.model_name,
            success_callbacks=None,  # Can be configured as needed
            failure_callbacks=None,  # Can be configured as needed
        )

    def convert_message_to_litellm_format(self, message: Message) -> Dict[str, str]:
        """Convert Message to LiteLLM format."""
        return convert_message_to_litellm_format(message)

    def parse_json_response(
        self, content: str, default_additional_question: str = ''
    ) -> tuple[str, str]:
        """
        Parse JSON response to extract response and additional_question.

        Args:
            content: The raw response content from LLM
            default_additional_question: Default additional question if parsing fails

        Returns:
            tuple: (parsed_content, additional_question)
        """
        try:
            # Handle JSON wrapped in ```json and ``` code blocks
            if '```json' in content and '```' in content:
                # Extract JSON from code block
                start_idx = content.find('```json') + 7
                end_idx = content.rfind('```')
                if start_idx > 6 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx].strip()
                    parsed_response = json.loads(json_content)
                else:
                    # Fallback to direct parsing
                    parsed_response = json.loads(content)
            else:
                # Direct JSON parsing
                parsed_response = json.loads(content)

            parsed_content = parsed_response.get('response', content)
            additional_question = parsed_response.get(
                'additional_question', default_additional_question
            )

            return parsed_content, additional_question

        except (json.JSONDecodeError, TypeError):
            # Fallback if JSON parsing fails
            return content, default_additional_question

    @message_exponential_backoff()
    def ask_base(
        self,
        messages: List[Dict[str, Any]],
        max_token: int = None,
        return_num: int = None,
        model: str = None,
        default_additional_question: str = '',
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """
        Base ask method that handles all types of messages (text, image, audio, video).

        Args:
            messages: List of message dictionaries in LiteLLM format
            max_token: Maximum tokens for response
            return_num: Number of response candidates
            model: Model name to use
            default_additional_question: Default additional question if JSON parsing fails
            *args, **kwargs: Additional arguments for completion

        Returns:
            Message with parsed response and additional_question
        """
        model = model or self.model_name
        max_token = max_token or self.default_max_tokens
        return_num = return_num or self.default_return_num

        # Use LiteLLM completion with the provided messages
        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_token,
            n=return_num,
            *args,
            **kwargs,
        )

        # Extract responses from all candidates
        contents = [
            response.choices[i].message.content for i in range(len(response.choices))
        ]

        # Parse JSON response using the common parsing method
        gist, additional_question = self.parse_json_response(
            contents[0], default_additional_question
        )

        return Message(
            role='assistant',
            gist=gist,
            additional_question=additional_question,
        )

    def call_llm(
        self,
        messages: List[Message],
        functions=None,
        function_call=None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        n: int = 1,
        process_id: int = 0,
        **kwargs,
    ) -> tuple[Dict[str, Any], int, int]:
        """
        Call LLM using LiteLLM with retry logic and error handling.
        Returns: (message_dict, error_code, total_tokens)
        """
        # Use instance defaults if not provided
        model = model or self.model_name
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature

        return call_llm(
            messages=messages,
            functions=functions,
            function_call=function_call,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            process_id=process_id,
            try_times=self.try_times,
            **kwargs,
        )

    def ask(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Ask method to be implemented by subclasses."""
        raise NotImplementedError(
            "The 'ask' method must be implemented in derived classes."
        )
