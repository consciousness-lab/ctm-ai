import time
from typing import Any, Callable, Dict, List, Type

import litellm
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..messengers import Message
from ..utils import message_exponential_backoff


def convert_messages_to_litellm_format(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert CTM messages to LiteLLM format."""
    result = []
    for m in messages:
        msg_text = m.content if m.content is not None else m.query
        if msg_text is None:
            continue
        result.append({'role': m.role, 'content': msg_text})
    return result


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def litellm_completion_request(
    messages: List[Message],
    functions=None,
    model: str = 'gpt-4o',
    max_tokens: int = 1024,
    temperature: float = 0.7,
    n: int = 1,
    **kwargs,
):
    """Make a completion request using LiteLLM with retry logic."""
    litellm_messages = convert_messages_to_litellm_format(messages)

    completion_kwargs = {
        'model': model,
        'messages': litellm_messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'n': n,
        **kwargs,
    }

    if functions:
        if isinstance(functions, dict):
            completion_kwargs['functions'] = [functions]
        else:
            completion_kwargs['functions'] = functions

    response = completion(**completion_kwargs)
    return response


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
        self.default_max_tokens = kwargs.get('max_tokens', 1024)
        self.default_temperature = kwargs.get('temperature', 0.7)

        # Configure LiteLLM settings
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Configure LiteLLM settings and callbacks."""
        # Set default model if not specified
        if not hasattr(litellm, 'model'):
            litellm.model = self.model_name

        # Configure logging and callbacks if needed
        # litellm.success_callback = ["lunary"] # Example callback
        # litellm.failure_callback = ["lunary"] # Example callback

    def convert_message_to_litellm_format(self, message: Message) -> Dict[str, str]:
        """Convert Message to LiteLLM format."""
        if message.content is None:
            raise ValueError('Message content cannot be None')

        return {'role': message.role, 'content': message.content}

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

        for attempt in range(self.try_times):
            time.sleep(15)
            try:
                response = litellm_completion_request(
                    messages,
                    functions=functions,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                    **kwargs,
                )

                message = response.choices[0].message
                total_tokens = response.usage.total_tokens

                if process_id == 0:
                    print(f'[process({process_id})] total tokens: {total_tokens}')

                message_dict = {
                    'role': message.role,
                    'content': message.content,
                }

                # Handle function calls if present
                if hasattr(message, 'function_call') and message.function_call:
                    message_dict['function_call'] = {
                        'name': message.function_call.name,
                        'arguments': message.function_call.arguments,
                    }

                    # Clean up function name if it has dots
                    if '.' in message_dict['function_call']['name']:
                        message_dict['function_call']['name'] = message_dict[
                            'function_call'
                        ]['name'].split('.')[-1]

                return message_dict, 0, total_tokens

            except Exception as e:
                print(
                    f'[process({process_id})] Attempt {attempt + 1}/{self.try_times} failed with error: {repr(e)}'
                )

        return (
            {
                'role': 'assistant',
                'content': f'[ERROR] Failed after {self.try_times} attempts.',
            },
            -1,
            0,
        )

    @message_exponential_backoff()
    def ask_standard(
        self,
        messages: List[Message],
        max_token: int = 300,
        return_num: int = 5,
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """Standard ask method for basic text completion using LiteLLM."""
        model = model or self.model_name

        litellm_messages = convert_messages_to_litellm_format(messages)

        response = completion(
            model=model,
            messages=litellm_messages,
            max_tokens=max_token,
            n=return_num,
            **kwargs,
        )

        gists = [response.choices[i].message.content for i in range(return_num)]
        return Message(
            role='assistant',
            content=gists[0],
            gist=gists[0],
            gists=gists,
        )

    def ask(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Ask method to be implemented by subclasses."""
        raise NotImplementedError(
            "The 'ask' method must be implemented in derived classes."
        )
