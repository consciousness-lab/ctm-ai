import concurrent.futures
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk, ChunkManager
from ..configs import ConsciousnessTuringMachineConfig
from ..graphs import ProcessorGraph
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import logging_func_with_count

if TYPE_CHECKING:
    from ..apis import BaseEnv

try:
    from ..apis import BaseEnv as _BaseEnv

    TOOLBENCH_AVAILABLE = True
except ImportError:
    TOOLBENCH_AVAILABLE = False
    _BaseEnv = None


class BaseConsciousnessTuringMachine(ABC):
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

    def reset(self) -> None:
        self.load_ctm()

    def load_ctm(self) -> None:
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []

        for processor_name in self.config.processors:
            self.processor_graph.add_node(
                processor_name=processor_name, processor_group_name=None
            )

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)

    def add_processor(
        self, processor_name: str, group_name: Optional[str] = None
    ) -> None:
        self.processor_graph.add_node(processor_name, group_name)

    def remove_processor(self, processor_name: str) -> None:
        self.processor_graph.remove_node(processor_name)

    def add_supervisor(self, name: str) -> None:
        self.supervisors.append(BaseSupervisor(name))

    def remove_supervisor(self, name: str) -> None:
        self.supervisors = [
            supervisor for supervisor in self.supervisors if supervisor.name != name
        ]

    def add_scorer(self, name: str) -> None:
        self.scorers.append(BaseScorer(name))

    def remove_scorer(self, name: str) -> None:
        self.scorers = [scorer for scorer in self.scorers if scorer.name != name]

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
        io_function: Optional['BaseEnv'] = None,
        memory_mode: Optional[bool] = None,
    ) -> Chunk:
        """Ask processor with support for both standard and tool processors"""
        if io_function and hasattr(processor, 'name'):
            # Tool processor
            return processor.ask(query, io_function, processor.name)
        else:
            # Standard processor
            return processor.ask(
                query=query,
                text=text,
                image=image,
                image_path=image_path,
                audio=audio,
                audio_path=audio_path,
                video_frames=video_frames,
                video_frames_path=video_frames_path,
                video_path=video_path,
                memory_mode=memory_mode,  # Pass memory mode to standard processor
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
        io_function: Optional['BaseEnv'] = None,
        memory_mode: Optional[bool] = None,  # Add memory mode support
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
                    memory_mode,  # Pass memory mode to each processor
                )
                for processor in self.processor_graph.nodes
            ]
            chunks = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        assert len(chunks) == len(self.processor_graph.nodes)
        return chunks

    @logging_func_with_count
    def ask_supervisor(
        self, query: str, chunk: Chunk
    ) -> Tuple[Union[str, None], float]:
        final_answer, score = self.supervisors[0].ask(query, chunk.gist)
        return final_answer, score

    @logging_func_with_count
    def uptree_competition(self, chunks: List[Chunk]) -> Chunk:
        chunk_manager = ChunkManager(chunks)
        return chunk_manager.uptree_competition()

    @logging_func_with_count
    def downtree_broadcast(self, chunk: Chunk) -> None:
        for processor in self.processor_graph.nodes:
            processor.update(chunk)

    @logging_func_with_count
    def link_form(
        self, chunks: List[Chunk], winning_chunk: Chunk, **input_kwargs
    ) -> None:
        """
        Form links between processors based on additional question processing.

        Args:
            chunks: List of chunks from previous processing
            winning_chunk: The winning chunk that contains additional_question
            **input_kwargs: All input parameters (text, image, audio, etc.)
        """
        additional_question = winning_chunk.additional_question
        # Use the same input parameters for processing additional question
        chunks = self.ask_processors(
            query=additional_question, **input_kwargs, memory_mode=False
        )

        for chunk in chunks:
            if chunk.confidence > 0.8:
                self.processor_graph.add_link(
                    processor1_name=winning_chunk.processor_name,
                    processor2_name=chunk.processor_name,
                )
            elif chunk.confidence < 0.2:
                self.processor_graph.remove_link(
                    processor1_name=winning_chunk.processor_name,
                    processor2_name=chunk.processor_name,
                )

    @logging_func_with_count
    def fuse_processor(self, chunks: List[Chunk]) -> List[Chunk]:
        linked_chunks: List[Tuple[Chunk, Chunk]] = []

        for chunk in chunks:
            src_chunk = chunk
            tgt_processor_names = self.processor_graph.get_neighbor_names(
                processor_name=src_chunk.processor_name
            )
            linked_chunks.extend(
                [
                    (src_chunk, chunk)
                    for chunk in chunks
                    if chunk.processor_name in tgt_processor_names
                ]
            )

        # Process each linked chunk pair and replace original chunks
        for chunk1, chunk2 in linked_chunks:
            processor1 = self.processor_graph.get_node(chunk1.processor_name)
            processor2 = self.processor_graph.get_node(chunk2.processor_name)

            if processor1 and processor2:
                # Add chunks to each other's processor memory
                processor1.update(chunk2)
                processor2.update(chunk1)

                # Re-ask both processors with their updated memory
                original_query = (
                    chunk1.gist
                    if chunk1.confidence > chunk2.confidence
                    else chunk2.gist
                )

                # Get updated chunks from processors
                updated_chunk1 = processor1.ask_with_memory(
                    query=original_query,
                    text=chunk1.gist,
                )
                updated_chunk2 = processor2.ask_with_memory(
                    query=original_query,
                    text=chunk2.gist,
                )

                # Replace original chunks with updated ones
                # Find and replace chunk1
                for i, chunk in enumerate(chunks):
                    if chunk.processor_name == chunk1.processor_name:
                        chunks[i] = updated_chunk1
                        break

                # Find and replace chunk2
                for i, chunk in enumerate(chunks):
                    if chunk.processor_name == chunk2.processor_name:
                        chunks[i] = updated_chunk2
                        break

        random.shuffle(chunks)
        return chunks

    @abstractmethod
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
        pass
