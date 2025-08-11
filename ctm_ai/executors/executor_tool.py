import json
from typing import Any, Dict, List, Optional, Union

import litellm

from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message
from ..utils import (
    call_llm,
    configure_litellm,
    convert_messages_to_litellm_format,
    message_exponential_backoff,
)


@BaseExecutor.register_executor('tool_executor')
class ToolExecutor(BaseExecutor):
    def __init__(
        self,
        name,
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

    def init_model(self, *args, **kwargs):
        super().init_model(*args, **kwargs)

    @message_exponential_backoff()
    def ask(
        self,
        messages: List[Message],
        api_manager: Any = None,
        function_name: str = None,
        query: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        messages = convert_messages_to_litellm_format(messages)
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            tools=self.api_func_info,
            tool_choice='auto',
        )
        response_message = response.choices[0].message
        breakpoint()
        retry_times = 3
        if response_message.content is not None and response_message.tool_calls is None:
            new_message = response_message.content
        elif (
            response_message.tool_calls is not None and response_message.content is None
        ):
            tool_call = response_message.tool_calls[0]
            func_name = getattr(tool_call.function, 'name', None) or self.function_name
            func_args = getattr(tool_call.function, 'arguments', '{}')

            if isinstance(func_args, dict):
                function_args = json.dumps(func_args, ensure_ascii=False)
            else:
                function_args = str(func_args) if func_args is not None else '{}'
            # function_args = response_message.tool_calls[0].function.arguments
            last_exc = None
            for i in range(retry_times):
                try:
                    new_message, status_code = self.api_manager.step(
                        action=func_name, input_str=function_args
                    )
                    if status_code in (0, 3):
                        break
                except Exception as e:
                    if i < retry_times - 1:
                        continue
                    else:
                        new_message = {
                            'error': f'tool execution failed: {type(e).__name__}: {e}',
                            'response': '',
                        }

        else:
            new_message = response_message.content or ''

        new_message = new_message['response']

        if isinstance(new_message, dict):
            new_message_str = json.dumps(new_message, ensure_ascii=False)
        else:
            new_message_str = str(new_message)

        prompt = f"""Regarding to the task: {query}, the answer is: {new_message}. If you think the answer is incorrect or incomplete to solve the task, you should generate a question that potentially can be answered by other tools. Your question should be just about what other tools you will need to better solve the task, nothing else about the task or original query should be included. Your question is:"""

        additional_question = litellm.completion(
            model=self.model_name, messages=[{'role': 'user', 'content': prompt}]
        )
        return Message(
            role='assistant',
            content=new_message_str,
            gist=new_message_str,
            additional_question=additional_question.choices[0].message.content
            if additional_question
            else None,
        )
