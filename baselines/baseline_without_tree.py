import concurrent.futures
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ctm_ai.chunks import Chunk
from ctm_ai.configs import ConsciousnessTuringMachineConfig
from ctm_ai.fusers import BaseFuser
from ctm_ai.graphs import ProcessorGraph
from ctm_ai.processors import BaseProcessor
from ctm_ai.scorers import BaseScorer
from ctm_ai.supervisors import BaseSupervisor
from ctm_ai.utils import logging_func


class ConsciousnessTuringMachineBaseline:
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
        self.fusers: List[BaseFuser] = []

        for group_name, processors in self.config.groups_of_processors.items():
            for processor_name in processors:
                self.processor_graph.add_node(
                    processor_name=processor_name, processor_group_name=group_name
                )

        self.add_supervisor(self.config.supervisor)

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

    def add_fuser(self, name: str) -> None:
        self.fusers.append(BaseFuser(name))

    def remove_fuser(self, name: str) -> None:
        self.fusers = [fuser for fuser in self.fusers if fuser.name != name]

    @staticmethod
    def ask_processor(
        processor: BaseProcessor,
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

    @logging_func
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

    @logging_func
    def ask_supervisor(self, query: str, gist: str) -> Tuple[str, float]:
        final_answer, score = self.supervisors[0].ask(query, gist)
        return final_answer, score

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
        prompt_header = (
            'Based on the following processor outputs across different modalities, please provide the most accurate and comprehensive answer.\n\n'
            '### Processor Outputs:\n'
        )

        processor_outputs = '\n'.join(
            [
                f'**Processor [{chunk.processor_name}] Output:**\n{chunk.gist}\n'
                for chunk in chunks
            ]
        )

        combined_gist = prompt_header + processor_outputs

        final_answer, confidence_score = self.ask_supervisor(query, combined_gist)

        return final_answer, confidence_score
