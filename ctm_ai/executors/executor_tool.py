import os
import sys

from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message

toolbench_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../ToolBench')
)
if toolbench_root not in sys.path:
    sys.path.insert(0, toolbench_root)

import time

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from toolbench.inference.Downstream_tasks.base_env import base_env


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    openai_key,
    messages,
    functions=None,
    function_call=None,
    model='gpt-4o',
):
    json_data = {
        'model': model,
        'messages': messages,
        'max_tokens': 1024,
        'temperature': 0.7,
    }
    if functions:
        json_data['functions'] = functions
    if function_call:
        json_data['function_call'] = function_call

    openai.api_key = openai_key
    response = openai.ChatCompletion.create(**json_data)
    return response


class ChatGPTFunction:
    def __init__(self, model='gpt-4o', openai_key='', try_times=3):
        self.model = model
        self.openai_key = openai_key
        self.TRY_TIMES = try_times

    def call(self, prompt, functions=None, function_call=None, process_id=0):
        messages = [{'role': 'user', 'content': prompt}]
        for attempt in range(self.TRY_TIMES):
            if attempt > 0:
                time.sleep(15)
            try:
                response = chat_completion_request(
                    self.openai_key,
                    messages,
                    functions=functions,
                    function_call=function_call,
                    model=self.model,
                )
                message = response['choices'][0]['message']
                total_tokens = response['usage']['total_tokens']

                if process_id == 0:
                    print(f'[process({process_id})] total tokens: {total_tokens}')

                if (
                    'function_call' in message
                    and '.' in message['function_call']['name']
                ):
                    message['function_call']['name'] = message['function_call'][
                        'name'
                    ].split('.')[-1]

                return message, 0, total_tokens

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
        io_function: base_env,
        openai_function_name: str = '',
        *args,
        **kwargs,
    ) -> dict:
        function = io_function.openai_name_reflect_all_info[openai_function_name][0]
        new_message, error_code, total_tokens = self.llm.call(
            messages, functions=function, function_call=openai_function_name
        )
        assert new_message['role'] == 'assistant'
        if 'content' in new_message.keys() and new_message['content'] != None:
            return Message(role='assistant', content=new_message['content'])

        if 'function_call' in new_message.keys():
            assert new_message['function_call']['name'] == openai_function_name
            function_input = new_message['function_call']['arguments']
            observation, status = io_function.step(
                action_name=openai_function_name, action_input=function_input
            )
            return Message(role='function', content=observation)
