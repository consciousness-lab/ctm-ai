from typing import List

from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message


@BaseExecutor.register_executor('tool_executor')
class ToolExecutor(BaseExecutor):
    def init_model(self, *args, **kwargs):
        """Initialize the model using the base class functionality."""
        super().init_model(*args, **kwargs)

    def ask(
        self,
        messages: List[Message],
        io_function,
        openai_function_name: str = '',
        *args,
        **kwargs,
    ) -> Message:
        """
        Ask method for tool execution using function calls.

        Args:
            messages: List of messages for the conversation
            io_function: Function interface for tool execution
            openai_function_name: Name of the OpenAI function to call

        Returns:
            Message containing the response or function result
        """
        function = io_function.openai_name_reflect_all_info[openai_function_name][0]

        # Use the base class LLM calling functionality
        new_message, error_code, total_tokens = self.call_llm(
            messages, functions=function, function_call=openai_function_name
        )

        assert new_message['role'] == 'assistant'

        # Handle regular content response
        if 'content' in new_message.keys() and new_message['content'] is not None:
            return Message(
                role='assistant',
                content=new_message['content'],
                gist=new_message['content'],
            )

        # Handle function call response
        if 'function_call' in new_message.keys():
            function_call_data = new_message['function_call']
            assert isinstance(function_call_data, dict)
            assert function_call_data['name'] == openai_function_name
            function_input = function_call_data['arguments']
            observation, status = io_function.step(openai_function_name, function_input)
            return Message(role='function', content=observation, gist=observation)

        return Message(
            role='assistant',
            content='[ERROR] No valid response from model',
            gist='[ERROR] No valid response from model',
        )
