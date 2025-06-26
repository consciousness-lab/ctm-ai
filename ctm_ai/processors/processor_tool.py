import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..executors import BaseExecutor
from ..messengers import BaseMessenger, Message
from ..scorers import BaseScorer
from .processor_base import BaseProcessor

import sys

toolbench_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../ToolBench")
)
if toolbench_root not in sys.path:
    sys.path.insert(0, toolbench_root)
from toolbench.inference.Downstream_tasks.base_env import base_env


@BaseProcessor.register_processor("tool_processor")
class ToolProcessor(BaseProcessor):
    REQUIRED_KEYS = ["OPENAI_API_KEY", "TOOLBENCH_KEY"]

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name="tool_messenger")

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name="tool_executor")

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name="tool_scorer")

    def ask(
        self,
        io_function: base_env,
        query: str,
        tool_name: str,
    ) -> Chunk:
        executor_messages = self.messenger.collect_executor_messages(
            query=query,
            io_function=io_function,
            tool_name=tool_name,
        )

        executor_output = self.executor.ask(
            messages=executor_messages,
            io_function=io_function,
            tool_name=tool_name,
        )

        scorer_messages = self.messenger.collect_scorer_messages(
            query=query,
            io_function=io_function,
            tool_name=tool_name,
            executor_output=executor_output,
        )

        scorer_output = self.scorer.ask(messages=scorer_messages)

        self.messenger.update(
            executor_output=executor_output,
            scorer_output=scorer_output,
        )

        chunk = self.merge_outputs_into_chunk(
            name=self.name, scorer_output=scorer_output, executor_output=executor_output
        )
        return chunk


def register_tool_processors(tool_names: List[str]):
    for tool_name in tool_names:
        processor_name = f"tool_processor_{tool_name}"
        BaseProcessor._processor_registry[processor_name] = ToolProcessor
