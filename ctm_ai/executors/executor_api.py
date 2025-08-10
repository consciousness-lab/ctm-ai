from typing import Any, List

from litellm import completion
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
        function_name: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)

        self.api_manager = api_manager
        self.function_name = function_name
        if self.api_manager and self.function_name:
            self.api_func_info = self.api_manager.funcs_to_all_info[self.function_name]
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

        openai_message = [{'role': messages[-1].role, 'content': messages[-1].content}]

        # Format tool for OpenAI API (needs type field)
        tool_definition = {'type': 'function', 'function': self.api_func_info}

        llm_kwargs = {
            'messages': openai_message,
            'model': model,
            'tools': [tool_definition],
        }
        new_message = (
            self.client.chat.completions.create(**llm_kwargs).choices[0].message
        )
        if new_message.content is not None:
            additional_question = self._generate_general_question(
                messages, new_message.content, self.api_manager, False
            )

            return Message(
                role='assistant',
                content=new_message.content,
                gist=new_message.content,
                additional_question=additional_question,
            )

        # Handle tool call response
        if new_message.tool_calls is not None:
            # tool_calls is a list, get the first tool call
            first_tool_call = new_message.tool_calls[0]
            function_call_data = first_tool_call.function.arguments

            # The arguments are typically a JSON string, need to parse if string
            if isinstance(function_call_data, str):
                import json

                function_arguments = json.loads(function_call_data)
            else:
                function_arguments = function_call_data

            # Generate additional question for tool calls
            additional_question = self._generate_general_question(
                messages, function_arguments, self.api_manager, True
            )

            return Message(
                role='assistant',
                content=str(function_arguments),
                gist=str(function_arguments),
                additional_question=additional_question,
            )

    def _generate_general_question(
        self,
        messages: List[Message],
        response: str,
        api_manager: Any,
        is_tool_call: bool,
    ) -> str:
        """Generate follow-up question for general responses"""
        # Get original query
        original_query = ''
        for msg in messages:
            if msg.content and msg.role == 'user':
                original_query = msg.content
                break

        if is_tool_call:
            prompt = f"Regarding the query: '{original_query}', the answer of the tool call is {response}. If you think the answer by the tool call is correct and can be used to answer the question, you should answer 'I have no other questions'. If you think the answer by the tool call is incorrect or incomplete, you should generate a question that potentially can be answered that you are not sure about. Note that the answer for the tool calls is a argument, not a specific answer to the question. So your decision should be based on the argument, not the answer. Your question is: "
        else:
            prompt = f"Regarding the query: '{original_query}', the response is {response}. If you think the answer by is incorrect or incomplete, you should generate a question that potentially can be answered by other API tools. Your question is: "

        messages = [
            Message(role='user', content=prompt),
        ]
        litellm_messages = [
            self.convert_message_to_litellm_format(msg) for msg in messages
        ]

        new_response = completion(
            model=self.model_name,
            messages=litellm_messages,
            max_tokens=100,
            n=1,
        )
        return new_response.choices[0].message.content
