from typing import Any, List

from openai import OpenAI

from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message
from ..utils.error_handler import message_exponential_backoff


@BaseExecutor.register_executor('api_executor')
class APIExecutor(BaseExecutor):
    def __init__(
        self,
        name: str,
        api_manager: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)

        self.api_manager = api_manager
        self.api_func_info = self.api_manager.funcs_to_all_info[name]
        self.client = OpenAI()

    def init_model(self, *args, **kwargs):
        """Initialize the model and functions for the tool executor."""
        self.system_prompt = "You are a helpful assistant with access to a variety of tools. Your task is to select the appropriate tool and use it to answer the user's query."
        super().init_model(*args, **kwargs)

    @message_exponential_backoff()
    def ask(
        self,
        messages: List[Message],
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        # Call LLM for tool selection and invocation
        model = model or self.model_name
        breakpoint()
        original_message = messages[-1]
        kwargs = {
            'messages': original_message,
            'model': model,
            'tools': self.api_func_info,
        }
        new_message = self.client.chat.completions.create(**kwargs).choices[0].message

        additional_question = self._generate_general_question(
            messages, new_message.content
        )
        # Handle normal content response (no tool used)
        if new_message.content is not None:
            return Message(
                role='assistant',
                content=new_message.content,
                gist=new_message.content,
                additional_question=additional_question,
            )

        # Handle tool call response
        if new_message.tool_calls is not None:
            function_call_data = new_message.tool_calls.function.arguments
            assert isinstance(function_call_data, dict)

            function_arguments = function_call_data['arguments']
            return Message(
                role='assistant',
                content=function_arguments,
                gist=function_arguments,
                additional_question=additional_question,
            )

    def _generate_general_question(self, messages: List[Message], response: str) -> str:
        """Generate follow-up question for general responses"""
        # Get original query
        original_query = ''
        for msg in messages:
            if msg.content and msg.role == 'user':
                original_query = msg.content
                break

        # Generate general follow-up questions
        general_questions = [
            f"Regarding '{original_query}', what specific aspects would you like to know more about?",
            'Would you like me to use search or calculation tools to get more information?',
            'Which part would you like me to explain in more detail?',
        ]

        return general_questions[0]
