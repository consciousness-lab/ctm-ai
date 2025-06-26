from typing import Callable, Dict, List, Optional, Tuple, Union
from .ctm_base import BaseConsciousnessTuringMachine
from ..chunks import Chunk
from ..processors import BaseProcessor
from ..configs import ConsciousnessTuringMachineConfig
from ..utils import logging_func_with_count
from ..fusers import BaseFuser
from ..graphs import ProcessorGraph
from ..processors import BaseProcessor, register_tool_processors
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
import sys
import os

import numpy as np
from numpy.typing import NDArray

toolbench_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../ToolBench")
)
if toolbench_root not in sys.path:
    sys.path.insert(0, toolbench_root)
from toolbench.inference.Downstream_tasks.base_env import base_env
from ..processors import ToolProcessor


class ToolConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(self, io_function, query, ctm_name: Optional[str] = None) -> None:
        super().__init__()
        self.config = (
            ConsciousnessTuringMachineConfig.from_ctm(ctm_name)
            if ctm_name
            else ConsciousnessTuringMachineConfig()
        )
        self.io_function = io_function
        self.query = query
        self.load_ctm()

    def __call__(
        self,
        query: str,
        io_function: base_env,
    ) -> Tuple[str, float]:
        return self.forward(
            query=query,
            io_function=io_function,
        )

    def load_ctm(self) -> None:
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []
        self.fusers: List[BaseFuser] = []

        standard_tool_names = self.io_fucntion.functions
        tool_names = [api[0] for api in standard_tool_names]
        register_tool_processors(tool_names)

        self.processor_graph = ProcessorGraph()

        for standard_tool_name in standard_tool_names:
            processor_name = f"tool_processor_{standard_tool_name}"
            self.processor_graph.add_node(
                processor_name=processor_name,
                processor_group_name="tools",
            )

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)
        self.add_fuser(self.config.fuser)
