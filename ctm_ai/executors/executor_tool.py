from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message
import json


import sys
import os

toolbench_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ToolBench"))
if toolbench_root not in sys.path:
    sys.path.insert(0, toolbench_root)

from toolbench.inference.Downstream_tasks.base_env import base_env
from toolbench.inference.server import get_rapidapi_response


@BaseExecutor.register_executor("too_executor")
class ToolBenchExecutor(BaseExecutor):
    def init_model(self, tool_env: base_env, *args, **kwargs):
        self.tool_env = tool_env 

    def ask(self, message: Message, tool_name: str = "", *args, **kwargs) -> dict:
        action_name = tool_name
        action_input = message.gist
        obs, status_code = self.tool_env._step(action_name=action_name, action_input=action_input)

        try:
            obs_json = json.loads(obs)
        except Exception:
            obs_json = {"error": "response format error", "response": obs}

        return {
            "result": obs_json.get("response", ""),
            "raw": obs_json,
            "status_code": status_code
        }
