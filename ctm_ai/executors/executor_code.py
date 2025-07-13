import os
from typing import Any, List

from litellm import completion

from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('code_executor')
class CodeExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model using LiteLLM for Qwen."""
        # Set default model to Qwen for code processing
        kwargs.setdefault('model', 'qwen-plus')
        super().init_model(*args, **kwargs)

        # Set up API key and base URL for Qwen
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

        # Configure LiteLLM for Qwen
        os.environ['QWEN_API_KEY'] = self.api_key
        os.environ['QWEN_BASE_URL'] = self.base_url

    def convert_message_to_param(self, message: Message) -> Any:
        if message.content is None:
            raise ValueError('Message content cannot be None')
        if message.role == 'system':
            return {'role': 'system', 'content': message.content}
        elif message.role == 'user':
            return {'role': 'user', 'content': message.content}
        elif message.role == 'assistant':
            return {'role': 'assistant', 'content': message.content}
        else:
            raise ValueError(f'Unsupported message role: {message.role}')

    def convert_message_to_litellm_format(self, message: Message) -> Any:
        """Convert a Message object to the format expected by LiteLLM."""
        return self.convert_message_to_param(message)

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
        """Ask method for code processing using LiteLLM with Qwen."""
        model = model or self.model_name

        # Convert messages to LiteLLM format
        litellm_messages = [
            self.convert_message_to_litellm_format(message) for message in messages
        ]

        # Use LiteLLM completion with Qwen
        response = completion(
            model=model,
            messages=litellm_messages,
            max_tokens=max_token,
            n=return_num,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        gists = [response.choices[i].message.content for i in range(return_num)]
        return Message(
            role='assistant',
            content=gists[0],
            gist=gists[0],
            gists=gists,
        )
