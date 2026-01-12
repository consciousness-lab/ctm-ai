import concurrent.futures
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk, ChunkManager
from ..configs import ConsciousTuringMachineConfig
from ..graphs import ProcessorGraph
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import logger, logging_func_with_count

if TYPE_CHECKING:
    pass

try:
    from ..apis import BaseEnv

    TOOLBENCH_AVAILABLE = True
except ImportError:
    TOOLBENCH_AVAILABLE = False
    BaseEnv = None


class BaseConsciousTuringMachine(ABC):
    def __init__(self, ctm_name: Optional[str] = None) -> None:
        super().__init__()
        self.config = (
            ConsciousTuringMachineConfig.from_ctm(ctm_name)
            if ctm_name
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

    def reset(self) -> None:
        self.load_ctm()

    def load_ctm(self) -> None:
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []

        for processor_name, processor_config in self.config.processors_config.items():
            self.processor_graph.add_node(
                processor_name=processor_name,
                processor_group_name=None,
                system_prompt=processor_config.get('system_prompt'),
                model=processor_config.get('model'),
            )

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)

    def add_processor(
        self, processor_name: str, group_name: Optional[str] = None
    ) -> None:
        """Add a processor to the CTM."""
        processor_config = self.config.processors_config.get(processor_name, {})
        self.processor_graph.add_node(
            processor_name=processor_name,
            processor_group_name=group_name,
            system_prompt=processor_config.get('system_prompt'),
            model=processor_config.get('model'),
        )

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
    ) -> Chunk:
        """Ask processor with support for both standard and tool processors"""
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
            query=additional_question,
            **input_kwargs,
        )

        for chunk in chunks:
            if chunk.relevance >= 0.8:
                logger.info(
                    f'Adding link between {winning_chunk.processor_name} and {chunk.processor_name}'
                )
                self.processor_graph.add_link(
                    processor1_name=winning_chunk.processor_name,
                    processor2_name=chunk.processor_name,
                )
            elif chunk.relevance <= 0.2:
                self.processor_graph.remove_link(
                    processor1_name=winning_chunk.processor_name,
                    processor2_name=chunk.processor_name,
                )

    @logging_func_with_count
    def fuse_processor(
        self, chunks: List[Chunk], query: str, **input_kwargs
    ) -> List[Chunk]:
        proc_map = {p.name: p for p in self.processor_graph.nodes}
        dirty: set[str] = set()  # processors whose memory got new info

        for chunk in chunks:
            q = chunk.additional_question
            if not q:
                continue

            for nbr in self.processor_graph.get_neighbor_names(chunk.processor_name):
                if nbr == chunk.processor_name:
                    proc_map[nbr].update(chunk)
                    dirty.add(nbr)
                    continue

                answer_chunk = proc_map[nbr].ask(
                    query=q,
                    is_fuse=True,
                    **input_kwargs,
                )
                input_kwargs['text'] += '(additional information: {})'.format(
                    answer_chunk.gist
                )
                dirty.add(chunk.processor_name)

        for idx, chunk in enumerate(chunks):
            if chunk.processor_name in dirty:
                p = proc_map[chunk.processor_name]
                chunks[idx] = p.ask(query=query, **input_kwargs)
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
