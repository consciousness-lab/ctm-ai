import json
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..configs import ConsciousTuringMachineConfig
from ..graphs import ProcessorGraph
from ..utils import logger, logging_func_with_count
from .ctm_base import BaseConsciousTuringMachine


class ConsciousTuringMachine(BaseConsciousTuringMachine):
    """Conscious Turing Machine.

    Args:
        ctm_name: Config name – loads ``ctm_conf/{ctm_name}_config.json``.
        api_manager: When supplied, tool processors are registered from the
            available functions in this manager (tool-use mode).
        num_additional_questions: Override the config value if given.
    """

    def __init__(
        self,
        ctm_name: Optional[str] = None,
        api_manager: Any = None,
        num_additional_questions: Optional[int] = None,
    ) -> None:
        self.api_manager = api_manager
        self.config = (
            ConsciousTuringMachineConfig.from_ctm(ctm_name)
            if ctm_name
            else ConsciousTuringMachineConfig()
        )
        if num_additional_questions is not None:
            self.config.num_additional_questions = num_additional_questions
        self.iteration_history: list = []
        self.detailed_log = None
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
    ) -> Tuple[str, float, str]:
        return self.forward(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
            api_manager=self.api_manager,
            instance_id=instance_id,
        )

    # ------------------------------------------------------------------
    # Processor loading
    # ------------------------------------------------------------------

    def load_ctm(self) -> None:
        """Load processors – from config or dynamically from api_manager."""
        if self.api_manager:
            self.processor_graph = ProcessorGraph()
            self._load_tool_processors()
        else:
            super().load_ctm()

    def _load_tool_processors(self) -> None:
        """Register tool processors from api_manager function list."""
        from ..processors import register_tool_processors

        openai_function_names = list(self.api_manager.function_names)
        register_tool_processors(openai_function_names)

        for func_name in openai_function_names:
            self.processor_graph.add_node(
                processor_name=func_name,
                processor_group_name='tool',
                model=getattr(self.config, 'model', 'gemini/gemini-2.0-flash-lite'),
                api_manager=self.api_manager,
                num_additional_questions=self.config.num_additional_questions,
                score_weights=self.config.score_weights,
            )

    # ------------------------------------------------------------------
    # CTM phases
    # ------------------------------------------------------------------

    @logging_func_with_count
    def go_down(
        self, winning_chunk: Chunk, chunks: List[Chunk], **input_kwargs
    ) -> None:
        logger.info(f'Going down with winning chunk: {winning_chunk.processor_name}')
        self.downtree_broadcast(winning_chunk)
        self.link_form(chunks, winning_chunk, **input_kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        api_manager: Any = None,
        instance_id: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Tuple[str, float, str]:
        """Run the iterative CTM loop.

        Returns:
            ``(answer, weight_score, parsed_answer)``
        """
        if api_manager is None:
            api_manager = self.api_manager

        input_params: dict = {
            'text': text,
            'image': image,
            'image_path': image_path,
            'audio': audio,
            'audio_path': audio_path,
            'video_frames': video_frames,
            'video_frames_path': video_frames_path,
            'video_path': video_path,
        }
        if api_manager is not None:
            input_params['api_manager'] = api_manager

        self.detailed_log = {
            'instance_id': instance_id,
            'initial_query': query,
            'iterations': [],
            'current_iteration': None,
        }

        self.iteration_history = []
        answer = ''
        weight_score = 0.0

        for i in range(self.config.max_iter_num):
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

            answer = winning_chunk.gist
            weight_score = winning_chunk.weight

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
                'winning_answer': winning_chunk.gist,
                'all_chunks': [
                    {
                        'processor_name': c.processor_name,
                        'weight': c.weight,
                        'relevance': c.relevance,
                        'confidence': c.confidence,
                        'surprise': c.surprise,
                    }
                    for c in chunks
                ],
            }
            self.iteration_history.append(iteration_info)

            if (
                i == self.config.max_iter_num - 1
                or weight_score >= self.config.output_threshold
            ):
                self.detailed_log['iterations'].append(
                    self.detailed_log['current_iteration']
                )
                self.detailed_log['current_iteration'] = None

                parsed_answer = self.parse_answer(answer=answer, query=query)

                self.detailed_log['final_answer'] = answer
                self.detailed_log['final_weight'] = weight_score
                self.detailed_log['parsed_answer'] = parsed_answer
                self._save_detailed_log()

                return answer, weight_score, parsed_answer

            self.go_down(winning_chunk, chunks, **input_params)
            self.fuse_processor(chunks, query, **input_params)

            self.detailed_log['iterations'].append(
                self.detailed_log['current_iteration']
            )

        parsed_answer = self.parse_answer(answer=answer, query=query)

        self.detailed_log['final_answer'] = answer
        self.detailed_log['final_weight'] = weight_score
        self.detailed_log['parsed_answer'] = parsed_answer
        self._save_detailed_log()

        return answer, weight_score, parsed_answer

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _save_detailed_log(self) -> None:
        if self.detailed_log is None or self.detailed_log.get('instance_id') is None:
            return

        ctm_name = self.config.ctm_name or 'default'
        output_dir = os.path.join('detailed_info', ctm_name)
        os.makedirs(output_dir, exist_ok=True)

        log_to_save = {
            k: v for k, v in self.detailed_log.items() if k != 'current_iteration'
        }

        instance_id = self.detailed_log['instance_id']
        output_path = os.path.join(output_dir, f'{instance_id}.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_to_save, f, indent=2, ensure_ascii=False)

        logger.info(f'Detailed log saved to {output_path}')
