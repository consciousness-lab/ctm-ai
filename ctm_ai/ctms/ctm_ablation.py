"""Ablation-enabled CTM.

Adds optional overrides on top of the standard ConsciousTuringMachine:
  - output_threshold_override: override config.output_threshold
  - link_form_threshold: override hardcoded 0.8 link-add threshold
  - enable_* flags: selectively disable CTM phases for component ablation
  - detailed_log_dir: where per-instance trajectories are saved
"""

import json
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..utils import logger, logging_func_with_count
from .ctm import ConsciousTuringMachine


class AblationCTM(ConsciousTuringMachine):
    """CTM with ablation knobs.

    All overrides are optional; when None we fall back to config / base defaults.
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
        output_threshold_override: Optional[float] = None,
        max_iter_override: Optional[int] = None,
        detailed_log_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            ctm_name=ctm_name,
            api_manager=api_manager,
            num_additional_questions=num_additional_questions,
        )
        self.enable_fusion = enable_fusion
        self.enable_uptree_competition = enable_uptree_competition
        self.enable_downtree_broadcast = enable_downtree_broadcast
        self.enable_link_form = enable_link_form
        self.enable_iteration = enable_iteration
        self.link_form_threshold = link_form_threshold
        self.output_threshold_override = output_threshold_override
        self.max_iter_override = max_iter_override
        self.detailed_log_dir = detailed_log_dir

        if output_threshold_override is not None:
            self.config.output_threshold = output_threshold_override
        if max_iter_override is not None:
            self.config.max_iter_num = max_iter_override

        self._iter_links_added = 0
        self._total_links_added = 0
        self._api_calls = 0
        self._parse_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}

    # ------------------------------------------------------------------
    # API call & token tracking
    # ------------------------------------------------------------------

    def get_usage_stats(self):
        """Aggregate usage stats from all processors (excluding parse)."""
        total = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}
        for proc in self.processor_graph.nodes:
            for k in total:
                total[k] += proc._usage_stats.get(k, 0)
        return total

    def get_parse_usage_stats(self):
        """Return parse-only usage stats."""
        return dict(self._parse_usage)

    def reset_usage_stats(self):
        """Reset all usage counters (call before each forward pass)."""
        for proc in self.processor_graph.nodes:
            proc._usage_stats = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}
        self._parse_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}

    # ------------------------------------------------------------------
    # Link formation with configurable thresholds
    # ------------------------------------------------------------------

    @logging_func_with_count
    def link_form(
        self, chunks: List[Chunk], winning_chunk: Chunk, **input_kwargs: Any
    ) -> None:
        """Form links and cache answers for fuse.

        Only asks non-winner processors (winner's own answer is useless
        for cross-modal exchange). If relevance >= threshold, add link
        AND cache the answer into the winner's fuse_history directly.
        Edges only grow (no link break).
        """
        import concurrent.futures

        form_t = (
            self.link_form_threshold if self.link_form_threshold is not None else 0.8
        )

        additional_questions = winning_chunk.additional_questions or []
        valid_questions = [q for q in additional_questions if q]
        if not valid_questions:
            return

        combined_query = 'Please answer the following questions:\n'
        for i, q in enumerate(valid_questions, 1):
            combined_query += f'{i}. {q}\n'

        # Only ask non-winner processors (saves 1 API call per iteration)
        proc_map = {p.name: p for p in self.processor_graph.nodes}
        w_name = winning_chunk.processor_name
        non_winner_procs = [p for p in self.processor_graph.nodes if p.name != w_name]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    lambda p: p.ask(query=combined_query, phase='link_form', **input_kwargs),
                    proc,
                )
                for proc in non_winner_procs
            ]
            question_chunks = [
                f.result() for f in concurrent.futures.as_completed(futures)
            ]
        question_chunks = [c for c in question_chunks if c is not None]

        if self.detailed_log is not None:
            current_iteration = self.detailed_log['current_iteration']
            for chunk in question_chunks:
                link_info = {
                    'processor_name': chunk.processor_name,
                    'query': chunk.executor_content or combined_query,
                    'answer': chunk.gist,
                    'relevance': chunk.relevance,
                }
                current_iteration['link_form_phase'].append(link_info)

        for chunk in question_chunks:
            c_name = chunk.processor_name

            if chunk.relevance >= form_t:
                already_linked = c_name in self.processor_graph.get_neighbor_names(w_name)
                self.processor_graph.add_link(
                    processor1_name=w_name,
                    processor2_name=c_name,
                )
                if not already_linked:
                    logger.info(
                        f'[ABLATION] Adding link (relevance={chunk.relevance:.3f} >= {form_t}) '
                        f'between {w_name} and {c_name}'
                    )
                    self._iter_links_added += 1
                    self._total_links_added += 1

                # Cache: add this processor's answer to winning processor's
                # fuse_history directly — no need to re-ask in fuse.
                if chunk.gist:
                    proc_map[w_name].add_fuse_history(
                        combined_query, chunk.gist, c_name
                    )

                    if self.detailed_log is not None:
                        current_iteration['fuse_phase'].append({
                            'from_processor': w_name,
                            'to_processor': c_name,
                            'query': combined_query,
                            'answer': chunk.gist,
                            'source': 'link_form_cache',
                        })

    # ------------------------------------------------------------------
    # Fuse: only winning processor answers other processors' questions
    # ------------------------------------------------------------------

    @logging_func_with_count
    def fuse_processor(
        self, chunks: List[Chunk], query: str, winning_chunk: Chunk = None, **input_kwargs
    ) -> None:
        """Efficient fuse: only the winning processor answers others' questions.

        link_form already cached non-winner → winner answers.
        Here we do the reverse: winner answers each linked non-winner's questions.
        """
        if winning_chunk is None:
            # Fallback to base implementation if no winner info
            return super().fuse_processor(chunks, query, **input_kwargs)

        proc_map = {p.name: p for p in self.processor_graph.nodes}
        w_name = winning_chunk.processor_name
        winner_proc = proc_map.get(w_name)
        if winner_proc is None:
            return

        for chunk in chunks:
            if chunk.processor_name == w_name:
                continue  # skip winner itself

            # Only fuse with linked processors
            if chunk.processor_name not in self.processor_graph.get_neighbor_names(w_name):
                continue

            additional_questions = chunk.additional_questions or []
            valid_questions = [q for q in additional_questions if q]
            if not valid_questions:
                continue

            combined_query = 'Please answer the following questions:\n'
            for i, q in enumerate(valid_questions, 1):
                combined_query += f'{i}. {q}\n'

            # Winner answers this processor's questions
            answer_chunk = winner_proc.ask(
                query=combined_query, phase='fuse', **input_kwargs
            )
            if answer_chunk is not None:
                proc_map[chunk.processor_name].add_fuse_history(
                    combined_query, answer_chunk.gist, w_name
                )

                if self.detailed_log is not None:
                    current_iteration = self.detailed_log['current_iteration']
                    current_iteration['fuse_phase'].append({
                        'from_processor': chunk.processor_name,
                        'to_processor': w_name,
                        'query': answer_chunk.executor_content or combined_query,
                        'answer': answer_chunk.gist,
                    })

    # ------------------------------------------------------------------
    # Go down (broadcast + link_form) with ablation flags
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
    # Forward with ablation flags (mirrors ConsciousTuringMachine.forward)
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
        self._api_calls = 0
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
                # Fallback: pick highest-weight chunk deterministically
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
                i == max_iters - 1
                or weight_score >= self.config.output_threshold
            )

            if is_final_iter:
                # Build iteration_info BEFORE returning (no link_form runs on last iter)
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

            # --- Downtree + link_form (ablation-aware) ---
            self.go_down(winning_chunk, chunks, **input_params)

            # --- Fusion (ablation-aware) ---
            if self.enable_fusion:
                self.fuse_processor(chunks, query, winning_chunk=winning_chunk, **input_params)
            else:
                logger.info('[ABLATION] Skipping fusion')

            # Record iteration AFTER link_form so link counts are correct
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

        # Fallback (should not normally reach here)
        parsed_answer = self.parse_answer(answer=answer, query=query)

        self.detailed_log['final_answer'] = answer
        self.detailed_log['final_weight'] = weight_score
        self.detailed_log['parsed_answer'] = parsed_answer
        self._save_detailed_log()

        return answer, weight_score, parsed_answer

    # ------------------------------------------------------------------
    # Save trajectories to a custom directory
    # ------------------------------------------------------------------

    def _save_detailed_log(self) -> None:
        if self.detailed_log is None or self.detailed_log.get('instance_id') is None:
            return

        output_dir = self.detailed_log_dir or 'detailed_info'
        os.makedirs(output_dir, exist_ok=True)

        log_to_save = {
            k: v for k, v in self.detailed_log.items() if k != 'current_iteration'
        }

        instance_id = self.detailed_log['instance_id']
        output_path = os.path.join(output_dir, f'{instance_id}.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_to_save, f, indent=2, ensure_ascii=False)

        logger.info(f'Detailed log saved to {output_path}')
