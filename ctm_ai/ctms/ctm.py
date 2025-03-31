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
            if confidence_score > self.config.output_threshold:
                return answer, confidence_score
            self.go_down(winning_chunk, chunks)
        return answer, confidence_score
