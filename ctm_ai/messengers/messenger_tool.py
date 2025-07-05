from typing import List, TypeVar

from ..apis.api_base import base_env as BaseEnv
from .message import Message
from .messenger_base import BaseMessenger

T = TypeVar('T', bound='BaseMessenger')
FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """You are an expert at using tools, you can use the tool {openai_function_name} to do the following task.
I will give you the api and function descriptions and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
Remember:
1. All the thought is short, at most in 5 sentence.
2. Follow the format, i.e,
Thought:
Action:
Action Input:
End Action
3. The Action: MUST be {openai_function_name}
Let's Begin!
Task description: {task_description}"""


@BaseMessenger.register_messenger('tool_messenger')
class ToolMessenger(BaseMessenger):
    def collect_executor_messages(
        self,
        query: str,
        io_function: BaseEnv,
        openai_function_name: str,
    ) -> List[Message]:
        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        system = system.replace('{openai_function_name}', openai_function_name)
        task_description = io_function.openai_name_reflect_all_info[
            openai_function_name
        ][1]
        system = system.replace(
            '{task_description}',
            task_description,
        )
        query_all = system + '\n' + query
        message = Message(role='user', query=query_all)
        self.executor_messages.append(message)

        return self.executor_messages

    def collect_scorer_messages(
        self,
        query: str,
        io_function: BaseEnv,
        openai_function_name: str,
        executor_output: Message,
    ) -> List[Message]:
        message = Message(
            role='assistant',
            gist=executor_output.gist,
        )
        self.scorer_messages.append(message)
        return self.scorer_messages
