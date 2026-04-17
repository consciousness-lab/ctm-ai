"""Ablation-enabled CTM.

Thin subclass of :class:`ConsciousTuringMachine` that exposes component and
threshold knobs for ablation studies. Core CTM logic (link_form / fuse /
usage tracking / detailed_log_dir) lives in the base class.

Overrides:
  - output_threshold_override, max_iter_override    configurable limits
  - link_form_threshold                             override LINK_FORM_THRESHOLD
  - enable_fusion / enable_uptree_competition /
    enable_downtree_broadcast / enable_link_form /
    enable_iteration                                selectively disable phases
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..utils import logger, logging_func_with_count
from .ctm import ConsciousTuringMachine


class AblationCTM(ConsciousTuringMachine):
    """CTM with ablation knobs.

    All overrides are optional; when ``None`` the base config / defaults apply.
    """

    def __init__(
        self,
        ctm_name: Optional[str] = None,
        api_manager: Any = None,
        num_additional_questions: Optional[int] = None,
        *,
        enable_fusion: bool = True,
        enable_uptree_competition: bool = True,
        enable_downtree_broadcast: bool = True,
        enable_link_form: bool = True,
        enable_iteration: bool = True,
        link_form_threshold: Optional[float] = None,
        link_form_ask_self: bool = False,
        output_threshold_override: Optional[float] = None,
        max_iter_override: Optional[int] = None,
        detailed_log_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            ctm_name=ctm_name,
            api_manager=api_manager,
            num_additional_questions=num_additional_questions,
            detailed_log_dir=detailed_log_dir,
        )
        self.enable_fusion = enable_fusion
        self.enable_uptree_competition = enable_uptree_competition
        self.enable_downtree_broadcast = enable_downtree_broadcast
        self.enable_link_form = enable_link_form
        self.enable_iteration = enable_iteration
        self.link_form_threshold = link_form_threshold
        self.link_form_ask_self = link_form_ask_self
        self.output_threshold_override = output_threshold_override
        self.max_iter_override = max_iter_override

        # Instance-level override — base class reads LINK_FORM_ASK_SELF via
        # self lookup, so setting as instance attr shadows the class attr.
        self.LINK_FORM_ASK_SELF = link_form_ask_self

        if output_threshold_override is not None:
            self.config.output_threshold = output_threshold_override
        if max_iter_override is not None:
            self.config.max_iter_num = max_iter_override

    # ------------------------------------------------------------------
    # Override link_form only to pick up a custom threshold.
    # ------------------------------------------------------------------

    @property
    def _active_link_form_threshold(self) -> float:
        return (
            self.link_form_threshold
            if self.link_form_threshold is not None
            else self.LINK_FORM_THRESHOLD
        )

    @logging_func_with_count
    def link_form(
        self, chunks: List[Chunk], winning_chunk: Chunk, **input_kwargs: Any
    ) -> None:
        # Temporarily shadow LINK_FORM_THRESHOLD for this call.
        original = self.LINK_FORM_THRESHOLD
        self.LINK_FORM_THRESHOLD = self._active_link_form_threshold
        try:
            super().link_form(chunks, winning_chunk, **input_kwargs)
        finally:
            self.LINK_FORM_THRESHOLD = original

    # ------------------------------------------------------------------
    # go_down: honor enable_downtree_broadcast / enable_link_form flags.
    # ------------------------------------------------------------------

    @logging_func_with_count
    def go_down(
        self, winning_chunk: Chunk, chunks: List[Chunk], **input_kwargs: Any
    ) -> None:
        if self.enable_downtree_broadcast:
            self.downtree_broadcast(winning_chunk)
        else:
            logger.info('[ABLATION] Skipping downtree_broadcast')
        if self.enable_link_form:
            self.link_form(chunks, winning_chunk, **input_kwargs)
        else:
            logger.info('[ABLATION] Skipping link_form')

    # ------------------------------------------------------------------
    # Forward: honor enable_iteration / enable_uptree_competition /
    # enable_fusion. All other behavior delegated to the base.
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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, float, str]:
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
            'ablation_flags': {
                'enable_fusion': self.enable_fusion,
                'enable_uptree_competition': self.enable_uptree_competition,
                'enable_downtree_broadcast': self.enable_downtree_broadcast,
                'enable_link_form': self.enable_link_form,
                'enable_iteration': self.enable_iteration,
                'link_form_threshold': self.link_form_threshold,
                'output_threshold_override': self.output_threshold_override,
            },
            'iterations': [],
            'current_iteration': None,
        }

        self.iteration_history = []
        self._total_links_added = 0
        self.reset_usage_stats()
        answer = ''
        weight_score = 0.0

        max_iters = 1 if not self.enable_iteration else self.config.max_iter_num

        for i in range(max_iters):
            self._iter_links_added = 0

            self.detailed_log['current_iteration'] = {
                'iteration': i + 1,
                'initial_phase': [],
                'winning_processor': None,
                'winning_weight': None,
                'link_form_phase': [],
                'fuse_phase': [],
            }

            chunks = self.ask_processors(query, **input_params)

            if self.enable_uptree_competition:
                winning_chunk = self.uptree_competition(chunks)
            else:
                winning_chunk = max(chunks, key=lambda c: c.weight)
                logger.info('[ABLATION] Skipping uptree_competition (argmax fallback)')

            answer = winning_chunk.gist
            weight_score = winning_chunk.weight

            self.detailed_log['current_iteration']['winning_processor'] = (
                winning_chunk.processor_name
            )
            self.detailed_log['current_iteration']['winning_weight'] = (
                winning_chunk.weight
            )

            is_final_iter = (
                i == max_iters - 1 or weight_score >= self.config.output_threshold
            )

            if is_final_iter:
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
                    'links_added': self._iter_links_added,
                }
                self.iteration_history.append(iteration_info)

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

            if self.enable_fusion:
                self.fuse_processor(
                    chunks, query, winning_chunk=winning_chunk, **input_params
                )
            else:
                logger.info('[ABLATION] Skipping fusion')

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
                'links_added': self._iter_links_added,
            }
            self.iteration_history.append(iteration_info)

            self.detailed_log['iterations'].append(
                self.detailed_log['current_iteration']
            )

        parsed_answer = self.parse_answer(answer=answer, query=query)

        self.detailed_log['final_answer'] = answer
        self.detailed_log['final_weight'] = weight_score
        self.detailed_log['parsed_answer'] = parsed_answer
        self._save_detailed_log()

        return answer, weight_score, parsed_answer
