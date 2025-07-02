import os
import sys
from typing import List, TypeVar

from toolbench.inference.Downstream_tasks.base_env import base_env

from .message import Message
from .messenger_base import BaseMessenger

toolbench_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../ToolBench')
)
if toolbench_root not in sys.path:
    sys.path.insert(0, toolbench_root)


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
        io_function: base_env,
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
        message = [
            Message(role='system', content=system),
            Message(role='user', query=query),
        ]
        self.executor_messages.append(message)

        return self.executor_messages

    def collect_scorer_messages(
        self,
        query: str,
        io_function: base_env,
        openai_function_name: str,
        executor_output: Message,
    ) -> List[Message]:
        message = Message(
            role='assistant',
            gist=executor_output.gist,
        )
        self.scorer_messages.append(message)
        return self.scorer_messages
