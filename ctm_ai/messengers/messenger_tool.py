from typing import Any, List, Optional, TypeVar

from ..apis import BaseEnv
from .message import Message
from .messenger_base import BaseMessenger

T = TypeVar('T', bound='BaseMessenger')


@BaseMessenger.register_messenger('tool_messenger')
class ToolMessenger(BaseMessenger):
    default_scorer_role: str = 'assistant'
    include_query_in_scorer: bool = True
    include_gists_in_scorer: bool = True

    default_executor_role: str = 'user'
    format_query_with_prefix: bool = False
    include_text_in_content: bool = False
    include_video_note: bool = False
    use_query_field: bool = False

    def collect_executor_messages(
        self,
        query: str,
        api_manager: Any = None,
        use_memory: bool = True,
        store_memory: bool = True,
        function_name: str = None,
        **kwargs: Any,
    ) -> List[Message]:
        content = self._build_executor_content(
            query=query, api_manager=api_manager, function_name=function_name, **kwargs
        )

        message_data = {
            'role': self.default_executor_role,
        }

        if self.use_query_field:
            message_data['query'] = content
        else:
            message_data['content'] = content

        current_message = Message(**message_data)

        messages_for_inference = []
        if self.system_prompt_message:
            messages_for_inference.append(self.system_prompt_message)

        if use_memory:
            messages_for_inference.extend(self.executor_messages)

        messages_for_inference.append(current_message)

        if store_memory:
            self.executor_messages.append(current_message)

        return messages_for_inference

    def _build_executor_content(
        self,
        query: str,
        api_manager: Any = None,
        function_name: str = None,
        **kwargs: Any,
    ) -> str:
        content = f'Task Description: {query}\n'

        content += f"""
You should utilize the other information in the context history and the information about the tool `{function_name}` to solve the task.
In the context history, there might have some answers to the task, you should utilize them to better solve and answer the task.

DECISION:
- First decide whether to call the tool `{function_name}`.
- If the tool helps even partially, CALL IT. Otherwise, answer directly.

OUTPUT PROTOCOL (MUST follow strictly):
- If you CALL the tool:
  - Return ONLY a function call via tool_calls.
  - Set assistant.content to null (no natural-language text).
  - Do NOT include any text explanation.
- If you DO NOT call the tool:
  - Return ONLY a natural-language answer in assistant.content.
  - Do NOT include tool_calls.
"""

        return content
