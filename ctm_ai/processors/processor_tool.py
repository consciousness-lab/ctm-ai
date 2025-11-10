from typing import Any, Dict, List

from .processor_base import BaseProcessor

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
{action_space}
Let's Begin!
"""


@BaseProcessor.register_processor('tool_processor')
class ToolProcessor(BaseProcessor):
    REQUIRED_KEYS = ['OPENAI_API_KEY']

    def _init_info(self, *args: Any, **kwargs: Any) -> None:
        self.system_prompt = 'You are an expert in tool calling.'

    def build_executor_messages(
        self,
        query: str,
        api_manager: Any,
        openai_function_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        system = system.replace('{openai_function_name}', openai_function_name)
        action_space = api_manager.openai_name_reflect_all_info[openai_function_name][1]
        system = system.replace(
            '{action_space}',
            action_space,
        )
        query_all = system + '\n' + query
        return [{'role': 'user', 'content': f'Query: {query_all}\n'}]
