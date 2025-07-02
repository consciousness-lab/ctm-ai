import concurrent.futures
import os
import sys
from typing import List, Optional, Tuple

from toolbench.inference.Downstream_tasks.base_env import base_env

from ..chunks import Chunk
from ..configs import ConsciousnessTuringMachineConfig
from ..fusers import BaseFuser
from ..graphs import ProcessorGraph
from ..processors import BaseProcessor, register_tool_processors
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import logging_func, logging_func_with_count
from .ctm_base import BaseConsciousnessTuringMachine

toolbench_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../ToolBench')
)
if toolbench_root not in sys.path:
    sys.path.insert(0, toolbench_root)


class ToolConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(self, io_function, query, ctm_name: Optional[str] = None) -> None:
        if hasattr(io_function, 'openai_function_names'):
            print(f'openai_function_names: {io_function.openai_function_names}')

        self.io_function = io_function
        self.query = query
        self.config = ConsciousnessTuringMachineConfig()

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
        if not hasattr(self, 'io_function') or self.io_function is None:
            raise ValueError('io_function has not been properly initialized!')
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []
        self.fusers: List[BaseFuser] = []

        openai_function_names = self.io_function.openai_function_names
        openai_function_names = [name for name in openai_function_names]
        register_tool_processors(openai_function_names)

        self.processor_graph = ProcessorGraph()
        for openai_function_name in openai_function_names:
            processor_name = openai_function_name
            self.processor_graph.add_node(
                processor_name=processor_name,
                processor_group_name='tools',
            )

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)
        self.add_fuser(self.config.fuser)

    @staticmethod
    def ask_processor(
        processor: BaseProcessor,
        query: str,
        io_function: base_env,
    ) -> Chunk:
        return processor.ask(
            query=query, io_function=io_function, openai_function_name=processor.name
        )

    @logging_func
    def ask_processors(self, query: str, io_function: base_env) -> List[Chunk]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.ask_processor, processor, query, io_function)
                for processor in self.processor_graph.nodes
            ]
            chunks = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        assert len(chunks) == len(self.processor_graph.nodes)
        return chunks

    @logging_func_with_count
    def go_up(
        self,
        query: str,
        io_function=None,
    ) -> Tuple[Chunk, List[Chunk]]:
        chunks = self.ask_processors(
            query,
            io_function,
        )
        chunks = self.fuse_processor(chunks)
        winning_chunk = self.uptree_competition(chunks)
        return winning_chunk, chunks

    @logging_func_with_count
    def go_down(self, winning_chunk: Chunk, chunks: List[Chunk]) -> None:
        self.downtree_broadcast(winning_chunk)
        self.link_form(chunks)

    def forward(
        self,
        query: str,
        io_function=None,
    ) -> Tuple[str, float]:
        """Forward pass supporting both standard and tool-based processing"""
        # Use provided io_function or fall back to instance io_function

        for _ in range(self.config.max_iter_num):
            winning_chunk, chunks = self.go_up(
                query,
                io_function,
            )
            answer, confidence_score = self.ask_supervisor(query, winning_chunk)
            confidence_score = 0
            if confidence_score > self.config.output_threshold:
                return answer, confidence_score
            self.go_down(winning_chunk, chunks)
        return answer, confidence_score
