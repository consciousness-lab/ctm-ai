from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message
import json


import sys
import os

toolbench_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../ToolBench")
)
if toolbench_root not in sys.path:
    sys.path.insert(0, toolbench_root)

from toolbench.inference.Downstream_tasks.base_env import base_env


@BaseExecutor.register_executor("tool_executor")
class ToolExecutor(BaseExecutor):
    def init_model(self, api_info, io_function: base_env, *args, **kwargs):
        self.io_function = io_function
        self.api_info = api_info

    def ask(self, message: Message, tool_name: str = "", *args, **kwargs) -> dict:
        action_name = tool_name
        action_input = message.gist
        obs, status_code = self.tool_env._step(
            action_name=action_name, action_input=action_input
        )

        try:
            obs_json = json.loads(obs)
        except Exception:
            obs_json = {"error": "response format error", "response": obs}

        return {
            "result": obs_json.get("response", ""),
            "raw": obs_json,
            "status_code": status_code,
        }
