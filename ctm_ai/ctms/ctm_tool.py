import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..apis import BaseEnv
from ..chunks import Chunk
from ..configs import ConsciousTuringMachineConfig
from ..graphs import ProcessorGraph
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import logging_func_with_count
from .ctm_base import BaseConsciousTuringMachine


class ToolConsciousTuringMachine(BaseConsciousTuringMachine):
    def __init__(
        self,
        ctm_name: Optional[str] = None,
        api_manager: BaseEnv = None,
    ) -> None:
        self.api_manager = api_manager
        self.config = ConsciousTuringMachineConfig()

        self.load_ctm()

    def __call__(
        self,
        query: str,
    ) -> Tuple[str, float]:
        return self.forward(
            query,
            self.api_manager,
        )

    def load_ctm(self) -> None:
        """Load CTM with support for both standard processors and tool processors"""
        # First, run the base class's loading logic to handle standard processors
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []
        # Then, add the specialized logic for this subclass to load tool processors
        if self.api_manager:
            self._load_tool_processors()

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)

    def _load_tool_processors(self) -> None:
        """Load TOOL processors."""

        from ..processors import register_tool_processors

        openai_function_names = [name for name in self.api_manager.function_names]
        register_tool_processors(openai_function_names)
        for openai_function_name in openai_function_names:
            processor_name = openai_function_name
            self.processor_graph.add_node(
                processor_name=processor_name,
                processor_group_name='tool',
                model=getattr(self.config, 'model', 'gemini/gemini-2.0-flash-lite'),
                api_manager=self.api_manager,  # Pass api_manager to the processor
            )

    @staticmethod
    def ask_processor(
        processor,
        query: str,
        text: Optional[str] = None,
        api_manager: BaseEnv = None,
        use_memory: bool = True,
        store_memory: bool = True,
    ) -> Chunk:
        """Ask processor with support for both standard and tool processors"""
        return processor.ask(
            query=query,
            text=text,
            api_manager=api_manager,
            use_memory=use_memory,
            store_memory=store_memory,
        )

    @logging_func_with_count
    def ask_processors(
        self, query: str, api_manager: BaseEnv = None, **kwargs
    ) -> List[Chunk]:
        if api_manager is None:
            api_manager = self.api_manager

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for processor in self.processor_graph.nodes:
                # Check if this is a BFCL processor that needs api_manager
                if (
                    hasattr(processor, 'name')
                    and processor.name in self.api_manager.function_names
                ):
                    # For BFCL processors, pass api_manager
                    future = executor.submit(
                        processor.ask, query=query, api_manager=api_manager, **kwargs
                    )
                else:
                    # For standard processors, use the base class method
                    future = executor.submit(
                        processor.ask, query=query, api_manager=api_manager, **kwargs
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
        api_manager: BaseEnv,
    ) -> Tuple[str, float]:
        """Forward pass supporting both standard and tool-based processing"""
        # Collect all input parameters for reuse
        input_params = {
            'api_manager': api_manager,
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
