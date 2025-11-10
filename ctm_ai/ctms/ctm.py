from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import random

from ..chunks import Chunk
from ..configs import ConsciousTuringMachineConfig
from ..utils import (
    logger,
    logging_func_with_count,
    log_forward_iteration,
    log_go_up_iteration,
)
from .ctm_base import BaseConsciousTuringMachine

if TYPE_CHECKING:
    from ..apis import BaseEnv

try:
    from ..apis import BaseEnv

    TOOLBENCH_AVAILABLE = True
except ImportError:
    TOOLBENCH_AVAILABLE = False
    BaseEnv = None


class ConsciousTuringMachine(BaseConsciousTuringMachine):
    def __init__(
        self, ctm_name: Optional[str] = None, api_manager: Optional["BaseEnv"] = None
    ) -> None:
        self.api_manager = api_manager
        self.config = (
            ConsciousTuringMachineConfig.from_ctm(ctm_name)
            if ctm_name != "toolbench"
            else ConsciousTuringMachineConfig()
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
        # First, run the base class's loading logic to handle standard processors
        super().load_ctm()

        # Then, add the specialized logic for this subclass to load tool processors
        if self.api_manager and TOOLBENCH_AVAILABLE:
            self._load_tool_processors()

    def _load_tool_processors(self) -> None:
        """Load tool processors if ToolBench is available."""
        if not TOOLBENCH_AVAILABLE:
            return

        from ..processors import register_tool_processors

        openai_function_names = [
            name for name in self.api_manager.openai_function_names
        ]
        register_tool_processors(openai_function_names)
        for openai_function_name in openai_function_names:
            processor_name = openai_function_name
            self.processor_graph.add_node(
                processor_name=processor_name, processor_group_name="tools"
            )

    @logging_func_with_count
    @log_go_up_iteration
    def go_up(self, query: str, **input_kwargs) -> Tuple[Chunk, List[Chunk]]:
        chunks = self.ask_processors(query, **input_kwargs)
        chunks = self.fuse_processor(chunks, query, **input_kwargs)
        winning_chunk = self.uptree_competition(chunks)
        return winning_chunk, chunks

    @logging_func_with_count
    def go_down(
        self, winning_chunk: Chunk, chunks: List[Chunk], **input_kwargs
    ) -> None:
        logger.info(f"Going down with winning chunk: {winning_chunk.processor_name}")
        self.downtree_broadcast(winning_chunk)
        self.link_form(chunks, winning_chunk, **input_kwargs)

    @log_forward_iteration
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
            "text": text,
            "image": image,
            "image_path": image_path,
            "audio": audio,
            "audio_path": audio_path,
            "video_frames": video_frames,
            "video_frames_path": video_frames_path,
            "video_path": video_path,
        }

        for i in range(self.config.max_iter_num):
            winning_chunk, chunks = self.go_up(
                query,
                **input_params,
            )
            answer, confidence_score = self.ask_supervisor(query, winning_chunk)
            if i == self.config.max_iter_num - 1:
                return answer, confidence_score
            self.go_down(winning_chunk, chunks, **input_params)
        return answer, confidence_score

    # Convenience method for tool-only usage (backward compatibility)
    def forward_tool(
        self,
        query: str,
        api_manager: "BaseEnv",
    ) -> Tuple[str, float]:
        """Forward pass for tool-only processing (backward compatibility)"""
        if not TOOLBENCH_AVAILABLE:
            raise ImportError(
                "ToolBench is not available. Please install ToolBench to use tool functionality."
            )

        return self.forward(query=query)
