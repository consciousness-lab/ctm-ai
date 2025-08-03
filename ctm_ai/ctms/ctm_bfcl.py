from typing import Dict, Any, List, Optional, Tuple

from ..chunks import Chunk
from ..configs import ConsciousTuringMachineConfig
from ..utils import logging_func_with_count
from .ctm_base import BaseConsciousTuringMachine
from ..apis import BFCLManager

from ..graphs import ProcessorGraph
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
import concurrent.futures


class BFCLConsciousTuringMachine(BaseConsciousTuringMachine):
    def __init__(
        self,
        ctm_name: Optional[str] = None,
        inference_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.io_function = BFCLManager(inference_data)
        self.config = ConsciousTuringMachineConfig()

        self.load_ctm()

    def __call__(
        self,
        query: str,
    ) -> Tuple[str, float]:
        breakpoint()
        return self.forward(
            query,
            self.io_function,
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

    def _load_bfcl_processors(self) -> None:
        """Load BFCL processors."""

        from ..processors import register_api_processors

        openai_function_names = [name for name in self.io_function.function_names]
        register_api_processors(openai_function_names)
        for openai_function_name in openai_function_names:
            processor_name = openai_function_name
            self.processor_graph.add_node(
                processor_name=processor_name, processor_group_name="bfcl"
            )
            breakpoint()

    @logging_func_with_count
    def ask_processors(
        self, query: str, io_function: BFCLManager = None, **kwargs
    ) -> List[Chunk]:
        """Override ask_processors to handle io_function parameter for BFCL processors"""
        if io_function is None:
            io_function = self.io_function

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for processor in self.processor_graph.nodes:
                # Check if this is a BFCL processor that needs io_function
                if (
                    hasattr(processor, "name")
                    and processor.name in self.io_function.function_names
                ):
                    # For BFCL processors, pass io_function
                    future = executor.submit(
                        processor.ask, query=query, io_function=io_function, **kwargs
                    )
                else:
                    # For standard processors, use the base class method
                    future = executor.submit(
                        self.ask_processor, processor, query, **kwargs
                    )
                futures.append(future)

            chunks = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        assert len(chunks) == len(self.processor_graph.nodes)
        return chunks

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
            "io_function": io_function,
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
