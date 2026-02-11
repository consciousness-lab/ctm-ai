from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..configs import ConsciousTuringMachineConfig
from ..utils import logger, logging_func_with_count
from .ctm_base import BaseConsciousTuringMachine


class ConsciousTuringMachine(BaseConsciousTuringMachine):
    def __init__(self, ctm_name: Optional[str] = None) -> None:
        self.config = ConsciousTuringMachineConfig.from_ctm(ctm_name)
        self.iteration_history = []
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
        super().load_ctm()

    @logging_func_with_count
    def go_down(
        self, winning_chunk: Chunk, chunks: List[Chunk], **input_kwargs
    ) -> None:
        logger.info(f'Going down with winning chunk: {winning_chunk.processor_name}')
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

        self.iteration_history = []

        for i in range(self.config.max_iter_num):
            chunks = self.ask_processors(query, **input_params)

            winning_chunk = self.uptree_competition(chunks)

            answer, weight_score = winning_chunk.gist, winning_chunk.weight

            iteration_info = {
                'iteration': i + 1,
                'winning_processor': winning_chunk.processor_name,
                'winning_weight': winning_chunk.weight,
                'all_chunks': [
                    {
                        'processor_name': c.processor_name,
                        'weight': c.weight,
                        'relevance': c.relevance,
                    }
                    for c in chunks
                ],
            }
            self.iteration_history.append(iteration_info)

            if (
                i == self.config.max_iter_num - 1
                or weight_score >= self.config.output_threshold
            ):
                parsed_answer = self.parse_answer(answer=answer, query=query)
                return answer, weight_score, parsed_answer

            self.downtree_broadcast(winning_chunk)

            self.link_form(chunks, winning_chunk, **input_params)

            self.fuse_processor(chunks, query, **input_params)

        return answer, weight_score, parsed_answer
