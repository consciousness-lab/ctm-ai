from typing import List, TypeVar

from ..apis import BaseEnv
from .message import Message
from .messenger_base import BaseMessenger

T = TypeVar('T', bound='BaseMessenger')
FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """You are an expert in solving tasks with or without the help of external tools.

I will give you a task and a tool description. Your job is to first decide whether the tool `{openai_function_name}` is necessary to solve the task according to the task description and the tool description. You should use the tool if it can solve part of the task.

At each step, you must:
- Provide a brief **Thought** (max 5 sentences) explaining your reasoning.
- If you decide to use the tool, provide an `Action`, `Action Input`, and `End Action`. The tool name must be `{openai_function_name}`.
- If you decide **not** to use the tool, explain your reasoning in the Thought and do not output Action.

Format (if you use the tool):
Thought:
Action: {openai_function_name}
Action Input: <valid JSON input>
End Action

Format (if not using the tool):
Thought: <reason why not using the tool>

Task description:
{task_description}
Let's Begin!
"""


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
