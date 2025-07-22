from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..configs import ConsciousnessTuringMachineConfig
from ..graphs import ProcessorGraph
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import logging_func_with_count
from .ctm_base import BaseConsciousnessTuringMachine

if TYPE_CHECKING:
    from ..apis import BaseEnv

try:
    from ..apis import BaseEnv as _BaseEnv

    TOOLBENCH_AVAILABLE = True
except ImportError:
    TOOLBENCH_AVAILABLE = False
    _BaseEnv = None


class ConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(
        self, ctm_name: Optional[str] = None, io_function: Optional['BaseEnv'] = None
    ) -> None:
        self.io_function = io_function
        self.config = (
            ConsciousnessTuringMachineConfig.from_ctm(ctm_name)
            if ctm_name
            else ConsciousnessTuringMachineConfig()
        )

        self.load_ctm()

    def __call__(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Tuple[str, float]:
        return self.forward(
            query,
            text,
            image,
            image_path,
            audio,
            audio_path,
            video_frames,
            video_frames_path,
            video_path,
        )

    def load_ctm(self) -> None:
        """Load CTM with support for both standard processors and tool processors"""
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []

        for processor_name in self.config.processors:
            self.processor_graph.add_node(
                processor_name=processor_name, processor_group_name=None
            )

        # Add tool processors if io_function is provided and ToolBench is available
        if self.io_function and TOOLBENCH_AVAILABLE:
            self._load_tool_processors()
        else:
            print('Warning: io_function is not provided or TOOLBENCH is not available.')

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)

    def _load_tool_processors(self) -> None:
        """Load tool processors if ToolBench is available."""
        if not TOOLBENCH_AVAILABLE:
            return

        tool_processors = [
            'tool_processor',
        ]

        for processor_name in tool_processors:
            self.add_processor(processor_name)

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
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Forward pass supporting both standard and tool-based processing"""
        # Collect all input parameters for reuse
        input_params = {
            'text': text,
            'image': image,
            'image_path': image_path,
            'audio': audio,
            'audio_path': audio_path,
            'video_frames': video_frames,
            'video_frames_path': video_frames_path,
            'video_path': video_path,
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

    # Convenience method for tool-only usage (backward compatibility)
    def forward_tool(
        self,
        query: str,
        io_function: 'BaseEnv',
    ) -> Tuple[str, float]:
        """Forward pass for tool-only processing (backward compatibility)"""
        if not TOOLBENCH_AVAILABLE:
            raise ImportError(
                'ToolBench is not available. Please install ToolBench to use tool functionality.'
            )

        return self.forward(query=query)
