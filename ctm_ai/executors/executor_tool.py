import os
from typing import Any
from ..messengers import Message
from ..utils import message_exponential_backoff
from .executor_base import BaseExecutor

@BaseExecutor.register_executor("tool_executor")
class ToolExecutor(BaseExecutor):
    def init_model(self, tool_name: str, *args, **kwargs):
        from toolbench.inference.server import get_rapidapi_response
        self.tool_name = tool_name
        self.get_response = get_rapidapi_response

    @message_exponential_backoff()
    def ask(self, message: Message, tool_name: str) -> dict:
        payload = {
            "tool_name": tool_name,
            "query": message.gist
        }
        return self.get_response(payload)
