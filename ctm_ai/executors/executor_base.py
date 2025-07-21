from typing import Any, Callable, Dict, List, Type

from ..messengers import Message
from ..utils import (
    ask_llm_standard,
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
        self.default_max_tokens = kwargs.get('max_tokens', 1024)
        self.default_temperature = kwargs.get('temperature', 0.7)

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

        gists = ask_llm_standard(
            messages=messages, model=model, max_tokens=max_token, n=return_num, **kwargs
        )
        breakpoint()

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
