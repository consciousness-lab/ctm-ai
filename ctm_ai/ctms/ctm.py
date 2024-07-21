import concurrent.futures
import random
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk, ChunkManager
from ..configs import ConsciousnessTuringMachineConfig
from ..fusers import BaseFuser
from ..graphs import ProcessorGraph
from ..processors import BaseProcessor
from ..utils import logging_func, logging_func_with_count


class ConsciousnessTuringMachine:
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
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Tuple[str, float]:
        return self.forward(query, text, image, audio, video_frames)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def load_ctm(self) -> None:
        self.processor_graph = ProcessorGraph()
        self.fusers: List[BaseFuser] = []

        for group_name, processors in self.config.groups_of_processors.items():
            for processor_name in processors:
                self.processor_graph.add_node(
                    processor_name=processor_name, processor_group_name=group_name
                )

        self.add_fuser(self.config.fuser)

    def add_fuser(self, name: str) -> None:
        self.fusers.append(BaseFuser(name))

    def remove_fuser(self, name: str) -> None:
        self.fusers = [fuser for fuser in self.fusers if fuser.name != name]

    @staticmethod
    def ask_processor(
        processor: BaseProcessor,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Chunk:
        return processor.ask(query, text, image, audio, video_frames)

    @logging_func
    def ask_processors(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> List[Chunk]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.ask_processor,
                    processor,
                    query,
                    text,
                    image,
                    audio,
                    video_frames,
                )
                for processor in self.processor_graph.nodes
            ]
            chunks = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        assert len(chunks) == len(self.processor_graph.nodes)
        return chunks

    @logging_func
    def uptree_competition(self, chunks: List[Chunk]) -> Chunk:
        chunk_manager = ChunkManager(chunks)
        return chunk_manager.uptree_competition()

    @logging_func
    def downtree_broadcast(self, chunk: Chunk) -> None:
        for processor in self.processor_graph.nodes:
            processor.update(chunk)

    @logging_func
    def link_form(self, chunks: List[Chunk]) -> None:
        chunk_manager = ChunkManager(chunks, self.config)
        interaction_matrix = chunk_manager.get_interaction_type_matrix()

        for i in range(len(interaction_matrix)):
            for j in range(i + 1, len(interaction_matrix)):
                interaction_type = interaction_matrix[i][j]

                if interaction_type != 0:
                    self.processor_graph.add_link(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                    )
                else:
                    self.processor_graph.remove_link(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                    )

    @logging_func
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

        for chunk1, chunk2 in linked_chunks:
            fused_chunk = self.fusers[0].fuse(chunk1, chunk2)
            chunks.append(fused_chunk)

        random.shuffle(chunks)
        return chunks

    @logging_func_with_count
    def forward(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Tuple[Chunk, List[Chunk]]:
        self.chunks = self.ask_processors(query, text, image, audio, video_frames)
        self.chunks = self.fuse_processor(self.chunks)
        self.winning_chunk = self.uptree_competition(self.chunks)
        answer = self.winning_chunk.gist
        return answer

    @logging_func_with_count
    def backward(self, feedback: Optional[str] = None) -> None:
        self.winning_chunk.feedback = feedback
        self.downtree_broadcast(self.winning_chunk)
        self.link_form(self.chunks)
