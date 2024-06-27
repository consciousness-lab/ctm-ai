import concurrent.futures
import random
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk, ChunkManager
from ..configs import BaseConsciousnessTuringMachineConfig
from ..fusers import BaseFuser
from ..graphs import ProcessorGraph
from ..processors import BaseProcessor
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import logging_func, logging_func_with_count


class BaseConsciousnessTuringMachine(object):
    def __init__(self, ctm_name: Optional[str] = None) -> None:
        super().__init__()
        if ctm_name:
            self.config = BaseConsciousnessTuringMachineConfig.from_ctm(ctm_name)
        else:
            self.config = BaseConsciousnessTuringMachineConfig()
        self.load_ctm()

    def __call__(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Tuple[str, float]:
        return self.forward(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )

    def load_ctm(self) -> None:
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []
        self.fusers: List[BaseFuser] = []
        for (
            group_name,
            processors,
        ) in self.config.groups_of_processors.items():
            for processor_name in processors:
                self.processor_graph.add_node(
                    processor_name=processor_name,
                    processor_group_name=group_name,
                )
        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)
        self.add_fuser(self.config.fuser)

    def add_supervisor(self, name: str) -> None:
        self.supervisors.append(BaseSupervisor(name))

    def remove_supervisor(self, name: str) -> None:
        for supervisor in self.supervisors:
            if supervisor.name == name:
                self.supervisors.remove(supervisor)

    def add_scorer(self, name: str) -> None:
        self.scorers.append(BaseScorer(name))

    def remove_scorer(self, name: str) -> None:
        for scorer in self.scorers:
            if scorer.name == name:
                self.scorers.remove(scorer)

    def add_fuser(self, name: str) -> None:
        self.fusers.append(BaseFuser(name))

    def remove_fuser(self, name: str) -> None:
        for fuser in self.fusers:
            if fuser.name == name:
                self.fusers.remove(fuser)

    @staticmethod
    def ask_processor(
        processor: BaseProcessor,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Chunk:
        chunk = processor.ask(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        return chunk

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
        assert len(chunks) == len(self.processor_graph)
        return chunks

    @logging_func
    def ask_supervisor(self, query: str, chunk: Chunk) -> Tuple[str, float]:
        final_answer, score = self.supervisors[0].ask(query, chunk.gist)
        return final_answer, score

    @logging_func
    def uptree_competition(self, chunks: List[Chunk]) -> Chunk:
        chunk_manager = ChunkManager(chunks)
        winning_chunk = chunk_manager.uptree_competition()
        return winning_chunk

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

                if interaction_type != 0:  # redundant or synergy
                    self.processor_graph.add_link(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                    )
                elif interaction_type == 0:  # uniqueness
                    self.processor_graph.remove_link(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                    )
        return

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
        for chunk_pair in linked_chunks:
            chunk1, chunk2 = chunk_pair
            fused_chunk = self.fusers[0].fuse(chunk1, chunk2)
            chunks.append(fused_chunk)
        random.shuffle(chunks)
        return chunks

    @logging_func_with_count
    def go_up(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Tuple[Chunk, List[Chunk]]:
        chunks = self.ask_processors(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        chunks = self.fuse_processor(chunks)
        winning_chunk = self.uptree_competition(chunks)
        return winning_chunk, chunks

    @logging_func_with_count
    def go_down(self, winning_chunk: 'Chunk', chunks: List['Chunk']) -> None:
        self.downtree_broadcast(winning_chunk)
        self.link_form(chunks)

    def forward(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Tuple[str, float]:
        for i in range(self.config.max_iter_num):
            winning_chunk, chunks = self.go_up(query, text, image, audio, video_frames)
            answer, confidence_score = self.ask_supervisor(query, winning_chunk)
            if confidence_score > self.config.output_threshold:
                return answer, confidence_score
            else:
                self.go_down(winning_chunk, chunks)
        return answer, confidence_score
