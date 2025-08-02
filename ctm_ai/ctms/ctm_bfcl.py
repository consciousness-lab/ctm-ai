from typing import Dict, Any, List, Optional, Tuple

from ..chunks import Chunk
from ..configs import ConsciousTuringMachineConfig
from ..utils import logging_func_with_count
from .ctm_base import BaseConsciousTuringMachine
from ..apis import BFCLManager

from ..graphs import ProcessorGraph
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor


class ConsciousTuringMachine(BaseConsciousTuringMachine):
    def __init__(
        self, ctm_name: Optional[str] = None, inference_data: Optional[Dict[str, Any]] = None
    ) -> None:
        self.io_function = BFCLManager(inference_data)
        self.config = (
            ConsciousTuringMachineConfig()
        )

        self.load_ctm()

    def __call__(
        self,
        query: str,
        io_function: BFCLManager,
    ) -> Tuple[str, float]:
        return self.forward(
            query,
            io_function,
        )

    def load_ctm(self) -> None:
        """Load CTM with support for both standard processors and tool processors"""
        # First, run the base class's loading logic to handle standard processors
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []
        # Then, add the specialized logic for this subclass to load tool processors
        if self.io_function:
            self._load_bfcl_processors()

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)

    def _load_tool_processors(self) -> None:
        """Load tool processors if ToolBench is available."""

        from ..processors import register_api_processors

        openai_function_names = [
            name for name in self.io_function.function_names
        ]
        register_api_processors(openai_function_names)
        for openai_function_name in openai_function_names:
            processor_name = openai_function_name
            self.processor_graph.add_node(
                processor_name=processor_name, processor_group_name='bfcl'
            )

    @logging_func_with_count
    def go_up(self, query: str, **input_kwargs) -> Tuple[Chunk, List[Chunk]]:
        for processor in self.processor_graph.nodes:
            processor.clear_memory()

        chunks = self.ask_processors(query, **input_kwargs)
        chunks = self.fuse_processor(chunks, query, **input_kwargs)
        winning_chunk = self.uptree_competition(chunks)
        return winning_chunk, chunks

    @logging_func_with_count
    def go_down(
        self, winning_chunk: Chunk, chunks: List[Chunk], **input_kwargs
    ) -> None:
        self.downtree_broadcast(winning_chunk)
        self.link_form(chunks, winning_chunk, **input_kwargs)

    def forward(
        self,
        query: str,
        io_function: BFCLManager,
    ) -> Tuple[str, float]:
        """Forward pass supporting both standard and tool-based processing"""
        # Collect all input parameters for reuse
        input_params = {
            'io_function': io_function,
        }

        for _ in range(self.config.max_iter_num):
            winning_chunk, chunks = self.go_up(
                query,
                **input_params,
            )
            answer, confidence_score = self.ask_supervisor(query, winning_chunk)
            if confidence_score > self.config.output_threshold:
                return answer, confidence_score
            self.go_down(winning_chunk, chunks, **input_params)
        return answer, confidence_score
