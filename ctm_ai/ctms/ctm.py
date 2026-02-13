import json
import os
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..configs import ConsciousTuringMachineConfig
from ..utils import logger, logging_func_with_count
from .ctm_base import BaseConsciousTuringMachine


class ConsciousTuringMachine(BaseConsciousTuringMachine):
    def __init__(
        self, ctm_name: Optional[str] = None, detailed_log_dir: str = 'detailed_info'
    ) -> None:
        self.config = ConsciousTuringMachineConfig.from_ctm(ctm_name)
        self.iteration_history = []
        self.detailed_log_dir = detailed_log_dir
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
        instance_id: Optional[str] = None,
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
            instance_id,
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
        instance_id: Optional[str] = None,
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

        # Initialize detailed log
        self.detailed_log = {
            'instance_id': instance_id,
            'initial_query': query,
            'iterations': [],
            'current_iteration': None,
        }

        self.iteration_history = []

        for i in range(self.config.max_iter_num):
            # Initialize current iteration log
            self.detailed_log['current_iteration'] = {
                'iteration': i + 1,
                'initial_phase': [],
                'winning_processor': None,
                'winning_weight': None,
                'link_form_phase': [],
                'fuse_phase': [],
            }

            chunks = self.ask_processors(query, **input_params)

            winning_chunk = self.uptree_competition(chunks)

            answer, weight_score = winning_chunk.gist, winning_chunk.weight

            # Update winning processor info in detailed log
            self.detailed_log['current_iteration']['winning_processor'] = (
                winning_chunk.processor_name
            )
            self.detailed_log['current_iteration']['winning_weight'] = (
                winning_chunk.weight
            )

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
                # Save current iteration before returning
                self.detailed_log['iterations'].append(
                    self.detailed_log['current_iteration']
                )
                self.detailed_log['current_iteration'] = None

                parsed_answer = self.parse_answer(answer=answer, query=query)

                # Add final answer to detailed log
                self.detailed_log['final_answer'] = answer
                self.detailed_log['final_weight'] = weight_score
                self.detailed_log['parsed_answer'] = parsed_answer

                # Save detailed log to file
                self._save_detailed_log()

                return answer, weight_score, parsed_answer

            self.downtree_broadcast(winning_chunk)

            self.link_form(chunks, winning_chunk, **input_params)

            self.fuse_processor(chunks, query, **input_params)

            # Save current iteration to iterations list
            self.detailed_log['iterations'].append(
                self.detailed_log['current_iteration']
            )

        # If we reach here, save the final state
        self.detailed_log['final_answer'] = answer
        self.detailed_log['final_weight'] = weight_score
        self.detailed_log['parsed_answer'] = parsed_answer
        self._save_detailed_log()

        return answer, weight_score, parsed_answer

    def _save_detailed_log(self) -> None:
        """Save detailed log to file"""
        if self.detailed_log is None or self.detailed_log['instance_id'] is None:
            return

        # Create detailed log folder if not exists
        output_dir = self.detailed_log_dir
        os.makedirs(output_dir, exist_ok=True)

        # Remove current_iteration from the log before saving
        log_to_save = {
            k: v for k, v in self.detailed_log.items() if k != 'current_iteration'
        }

        # Save to JSON file
        instance_id = self.detailed_log['instance_id']
        output_path = os.path.join(output_dir, f'{instance_id}.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_to_save, f, indent=2, ensure_ascii=False)

        logger.info(f'Detailed log saved to {output_path}')
