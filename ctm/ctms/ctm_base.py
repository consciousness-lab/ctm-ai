import concurrent.futures
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..configs import BaseConsciousnessTuringMachineConfig
from ..fusers import BaseFuser
from ..processors import BaseProcessor
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import (
    add_link_on_processor_graph,
    add_node_on_processor_graph,
    calc_chunk_sim,
    remove_link_on_processor_graph,
    remove_node_on_processor_graph,
)


class BaseConsciousnessTuringMachine(object):
    def __init__(self, ctm_name: Optional[str] = None) -> None:
        super().__init__()
        if ctm_name:
            self.config = BaseConsciousnessTuringMachineConfig.from_ctm(
                ctm_name
            )
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
        self.processor_graph: Dict[
            BaseProcessor, Set[BaseProcessor]
        ] = defaultdict(set)
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []
        self.fusers: List[BaseFuser] = []
        for (
            group_name,
            processors,
        ) in self.config.groups_of_processors.items():
            for processor_name in processors:
                self.add_processor(name=processor_name, group_name=group_name)
        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)

    def add_processor(
        self, name: str, group_name: Optional[str] = "default_group"
    ) -> None:
        self.processor_graph = add_node_on_processor_graph(
            processor_name=name,
            processor_group_name=group_name,
            processor_graph=self.processor_graph,
        )

    def remove_processor(self, name: str) -> None:
        self.processor_graph = remove_node_on_processor_graph(
            processor_name=name, processor_graph=self.processor_graph
        )

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
        verbose: Optional[bool] = False,
    ) -> Chunk:
        if verbose:
            print(processor.name, processor.group_name)

        chunk = processor.ask(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        return chunk

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
                for processor in self.processor_graph.keys()
            ]
            chunks = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
        assert len(chunks) == len(self.processor_graph)
        return chunks

    def ask_supervisor(self, query: str, chunk: Chunk) -> Tuple[str, float]:
        final_answer, score = self.supervisors[0].ask(query, chunk.gist)
        return final_answer, score

    def uptree_competition(self, chunks: List[Chunk]) -> Chunk:
        # Unpack processor outputs into lists for easier processing
        winning_chunks: List[Chunk] = []
        candidate_chunks: List[Chunk] = chunks
        for _ in range(len(chunks) - 1):
            for chunk1, chunk2 in zip(
                candidate_chunks[:-1], candidate_chunks[1:]
            ):
                winning_chunk = (
                    chunk1
                    if chunk1 > chunk2
                    else (
                        chunk2
                        if chunk1 < chunk2
                        else random.choice([chunk1, chunk2])
                    )
                )
                winning_chunks.append(winning_chunk)
            candidate_chunks = winning_chunks
            winning_chunks = []
        return candidate_chunks[0]

    def downtree_broadcast(self, chunk: Chunk) -> None:
        for processor in self.processor_graph.keys():
            processor.update(chunk)

    def link_form(self, chunks: List[Chunk]) -> None:
        sim = calc_chunk_sim(chunks)
        print(sim)
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                if sim[i][j] > 0.5:
                    self.processor_graph = add_link_on_processor_graph(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                        processor_graph=self.processor_graph,
                    )
                if sim[i][j] < 0.2:
                    self.processor_graph = remove_link_on_processor_graph(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                        processor_graph=self.processor_graph,
                    )
        return

    def processor_fuse(self, chunks: List[Chunk]) -> List[Chunk]:
        chunk_pairs: List[Tuple[Chunk, Chunk]] = []
        for chunk in chunks:
            src_chunk = chunk
            tgt_processor_names = self.processor_graph[
                src_chunk.processor_name
            ]
            chunk_pairs.extend(
                [
                    (src_chunk, chunk)
                    for chunk in chunks
                    if chunk.processor_name in tgt_processor_names
                ]
            )
        for chunk_pair in chunk_pairs:
            fused_chunk = self.fuser.fuse(chunk_pair)
            chunks.append(fused_chunk)
        random.shuffle(chunks)
        return chunks

    def forward(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
    ) -> Tuple[str, float]:
        answer_threshold = 0.5
        max_iter = 3

        for i in range(max_iter):
            print("start the {}-th iteration".format(i + 1))
            chunks = self.ask_processors(
                query=query,
                text=text,
                image=image,
                audio=audio,
                video_frames=video_frames,
            )
            chunks = self.processor_fuse(chunks)
            winning_chunk = self.uptree_competition(chunks)
            answer, confidence_score = self.ask_supervisor(
                query, winning_chunk
            )
            if confidence_score > answer_threshold:
                break
            else:
                self.downtree_broadcast(winning_chunk)
                self.link_form(chunks)
        return answer, confidence_score
