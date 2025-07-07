import os
import time
from typing import Dict, List

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message


def convert_messages_to_openai_format(messages: List[Message]) -> List[Dict[str, str]]:
    result = []
    for m in messages:
        msg_text = m.content if m.content is not None else m.query
        if msg_text is None:
            continue
        result.append({'role': m.role, 'content': msg_text})
    breakpoint()
    return result


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    openai_key,
    messages,
    functions=None,
    model='gpt-4o',
):
    openai_messages = convert_messages_to_openai_format(messages)
    breakpoint()
    json_data = {
        'model': model,
        'messages': openai_messages,
        'max_tokens': 1024,
        'temperature': 0.7,
    }
    if functions:
        if isinstance(functions, dict):
            json_data['functions'] = [functions]
        else:
            json_data['functions'] = functions
    # if function_call:
    #     if isinstance(function_call, str):
    #         json_data["function_call"] = {"name": function_call}
    #     else:
    #         json_data["function_call"] = function_call

    client = openai.OpenAI(api_key=openai_key)
    breakpoint()
    response = client.chat.completions.create(**json_data)
    return response


class ChatGPTFunction:
    def __init__(self, model='gpt-4o', openai_key='', try_times=3):
        self.model = model
        self.openai_key = openai_key
        self.TRY_TIMES = try_times

    def call(self, messages, functions=None, function_call=None, process_id=0):
        for attempt in range(self.TRY_TIMES):
            time.sleep(15)
            try:
                response = chat_completion_request(
                    self.openai_key,
                    messages,
                    functions=functions,
                    # function_call=function_call,
                    model=self.model,
                )
                message = response.choices[0].message
                total_tokens = response.usage.total_tokens

                if process_id == 0:
                    print(f'[process({process_id})] total tokens: {total_tokens}')

                message_dict = {
                    'role': message.role,
                    'content': message.content,
                }

                if hasattr(message, 'function_call') and message.function_call:
                    message_dict['function_call'] = {
                        'name': message.function_call.name,
                        'arguments': message.function_call.arguments,
                    }

                if (
                    'function_call' in message_dict
                    and '.' in message_dict['function_call']['name']
                ):
                    message_dict['function_call']['name'] = message_dict[
                        'function_call'
                    ]['name'].split('.')[-1]

                return message_dict, 0, total_tokens

            except Exception as e:
                print(
                    f'[process({process_id})] Attempt {attempt + 1}/{self.TRY_TIMES} failed with error: {repr(e)}'
                )

        return (
            {
                'role': 'assistant',
                'content': f'[ERROR] Failed after {self.TRY_TIMES} attempts.',
            },
            -1,
            0,
        )


@BaseExecutor.register_executor('tool_executor')
class ToolExecutor(BaseExecutor):
    def init_model(self, *args, **kwargs):
        self.llm = ChatGPTFunction(
            model='gpt-4o',
            openai_key=os.getenv('OPENAI_API_KEY', ''),
            try_times=3,
        )

    def ask(
        self,
        messages,
        io_function,
        openai_function_name: str = '',
        *args,
        **kwargs,
    ) -> Message:
        function = io_function.openai_name_reflect_all_info[openai_function_name][0]
        breakpoint()
        new_message, error_code, total_tokens = self.llm.call(
            messages, functions=function, function_call=openai_function_name
        )
        breakpoint()
        assert new_message['role'] == 'assistant'
        if 'content' in new_message.keys() and new_message['content'] is not None:
            return Message(
                role='assistant',
                content=new_message['content'],
                gist=new_message['content'],
            )

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
