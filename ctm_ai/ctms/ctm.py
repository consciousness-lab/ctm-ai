import concurrent.futures
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..configs import ConsciousnessTuringMachineConfig
from ..utils import logging_func_with_count
from .ctm_base import BaseConsciousnessTuringMachine


class ConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(self, ctm_name: Optional[str] = None) -> None:
        super().__init__()
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
        io_function=None,
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
            io_function,
        )

    def load_ctm(self) -> None:
        """Load CTM with support for both standard processors and tool processors"""
        super().load_ctm()

        # Add tool processors if io_function is provided and ToolBench is available
        if self.io_function and TOOLBENCH_AVAILABLE:
            self._load_tool_processors()

    def _load_tool_processors(self) -> None:
        """Load tool processors from io_function"""
        try:
            from ..processors import register_tool_processors

            openai_function_names = self.io_function.openai_function_names
            openai_function_names = [name for name in openai_function_names]
            register_tool_processors(openai_function_names)

            for openai_function_name in openai_function_names:
                processor_name = openai_function_name
                self.processor_graph.add_node(
                    processor_name=processor_name,
                    processor_group_name='tools',
                )
        except Exception as e:
            print(f'Warning: Failed to load tool processors: {e}')

    @staticmethod
    def ask_processor(
        processor,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        io_function=None,
    ) -> Chunk:
        """Ask processor with support for both standard and tool processors"""
        if io_function and hasattr(processor, 'name'):
            # Tool processor
            return processor.ask(query, io_function, processor.name)
        else:
            # Standard processor
            return processor.ask(
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

    @logging_func_with_count
    def ask_processors(
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
        io_function=None,
    ) -> List[Chunk]:
        """Ask all processors with support for both standard and tool processors"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.ask_processor,
                    processor,
                    query,
                    text,
                    image,
                    image_path,
                    audio,
                    audio_path,
                    video_frames,
                    video_frames_path,
                    video_path,
                    io_function,
                )
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
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        io_function=None,
    ) -> Tuple[Chunk, List[Chunk]]:
        chunks = self.ask_processors(
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
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Tuple[str, float]:
        for _ in range(self.config.max_iter_num):
            winning_chunk, chunks = self.go_up(
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
            answer, confidence_score = self.ask_supervisor(query, winning_chunk)
            confidence_score = 0
            if confidence_score > self.config.output_threshold:
                return answer, confidence_score
            self.go_down(winning_chunk, chunks)
        return answer, confidence_score
