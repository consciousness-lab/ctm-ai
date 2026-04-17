import concurrent.futures
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
        detailed_log_dir: Directory where per-instance trajectories are saved
            (defaults to ``detailed_info/``).
    """

    # Default link formation relevance threshold.
    LINK_FORM_THRESHOLD: float = 0.8

    # Whether the winning processor is also queried during link_form
    # (and whether a winner→winner edge may be added to the graph).
    # Default False — matches the efficient behavior of skipping the winner.
    LINK_FORM_ASK_SELF: bool = False

    def __init__(
        self,
        ctm_name: Optional[str] = None,
        api_manager: Any = None,
        num_additional_questions: Optional[int] = None,
        *,
        detailed_log_dir: Optional[str] = None,
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
        self.detailed_log_dir = detailed_log_dir

        # Usage / link counters (populated per forward call).
        self._iter_links_added = 0
        self._total_links_added = 0
        self._parse_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }

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
    # Usage tracking
    # ------------------------------------------------------------------

    def get_usage_stats(self):
        """Aggregate usage stats across processors (excludes parse step)."""
        total = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'api_calls': 0,
        }
        for proc in self.processor_graph.nodes:
            for k in total:
                total[k] += proc._usage_stats.get(k, 0)
        return total

    def get_parse_usage_stats(self):
        """Return parse-step token usage (parse is not counted as an api_call)."""
        return dict(self._parse_usage)

    def reset_usage_stats(self):
        """Reset all usage counters – call before each forward pass."""
        for proc in self.processor_graph.nodes:
            proc._usage_stats = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'api_calls': 0,
            }
        self._parse_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }

    # ------------------------------------------------------------------
    # link_form: only non-winner processors answer the winner's questions;
    # their answers are cached straight into the winner's fuse_history so
    # the fuse step does not have to re-ask.
    # ------------------------------------------------------------------

    @logging_func_with_count
    def link_form(
        self, chunks: List[Chunk], winning_chunk: Chunk, **input_kwargs: Any
    ) -> None:
        form_t = self.LINK_FORM_THRESHOLD

        additional_questions = winning_chunk.additional_questions or []
        valid_questions = [q for q in additional_questions if q]
        if not valid_questions:
            return

        combined_query = 'Please answer the following questions:\n'
        for i, q in enumerate(valid_questions, 1):
            combined_query += f'{i}. {q}\n'

        proc_map = {p.name: p for p in self.processor_graph.nodes}
        w_name = winning_chunk.processor_name
        ask_self = self.LINK_FORM_ASK_SELF
        procs_to_ask = (
            list(self.processor_graph.nodes)
            if ask_self
            else [p for p in self.processor_graph.nodes if p.name != w_name]
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    lambda p: p.ask(
                        query=combined_query, phase='link_form', **input_kwargs
                    ),
                    proc,
                )
                for proc in procs_to_ask
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
            already_linked = c_name in self.processor_graph.get_neighbor_names(
                w_name
            )
            passes_threshold = chunk.relevance >= form_t

            if passes_threshold:
                self.processor_graph.add_link(
                    processor1_name=w_name,
                    processor2_name=c_name,
                    allow_self=ask_self,
                )
                if not already_linked:
                    logger.info(
                        f'Adding link (relevance={chunk.relevance:.3f} >= {form_t}) '
                        f'between {w_name} and {c_name}'
                    )
                    self._iter_links_added += 1
                    self._total_links_added += 1

            # Cache the non-winner's answer into winner.fuse_history whenever
            # an edge exists between them — either newly formed this iter
            # (passes_threshold) or from a previous iter (already_linked).
            # This matches 09446d4 fuse semantics: fuse would fire over any
            # existing edge regardless of current relevance.
            if (passes_threshold or already_linked) and chunk.gist:
                proc_map[w_name].add_fuse_history(
                    combined_query, chunk.gist, c_name
                )

                if self.detailed_log is not None:
                    current_iteration['fuse_phase'].append(
                        {
                            'from_processor': w_name,
                            'to_processor': c_name,
                            'query': combined_query,
                            'answer': chunk.gist,
                            'source': 'link_form_cache',
                            'relevance': chunk.relevance,
                            'edge_pre_existing': already_linked,
                        }
                    )

    # ------------------------------------------------------------------
    # fuse_processor: only the winning processor answers the linked
    # non-winners' questions. (link_form already cached the reverse
    # direction.)
    # ------------------------------------------------------------------

    @logging_func_with_count
    def fuse_processor(
        self,
        chunks: List[Chunk],
        query: str,
        winning_chunk: Chunk = None,
        **input_kwargs: Any,
    ) -> None:
        if winning_chunk is None:
            return

        proc_map = {p.name: p for p in self.processor_graph.nodes}
        w_name = winning_chunk.processor_name

        # Iterate non-winner chunks; for each, ask ALL of its linked neighbors
        # (winner AND other linked non-winners) to answer its follow-up
        # questions. This restores the non-winner ↔ non-winner information flow
        # that existed in the original CTM and was missing from the winner-only
        # optimization. Winner's own follow-ups are already answered via
        # link_form caching, so we skip chunk = winner.

        # Pass 1: materialize the (c_name, nbr, nbr_proc, combined_query) task
        # list in the same order the original nested for-loops produced, so
        # downstream fuse_history / detailed_log append order is unchanged.
        tasks = []
        for chunk in chunks:
            c_name = chunk.processor_name
            if c_name == w_name:
                continue

            neighbors = self.processor_graph.get_neighbor_names(c_name)
            if not neighbors:
                continue

            additional_questions = chunk.additional_questions or []
            valid_questions = [q for q in additional_questions if q]
            if not valid_questions:
                continue

            combined_query = 'Please answer the following questions:\n'
            for i, q in enumerate(valid_questions, 1):
                combined_query += f'{i}. {q}\n'

            for nbr in neighbors:
                if nbr == c_name:
                    continue
                nbr_proc = proc_map.get(nbr)
                if nbr_proc is None:
                    continue
                tasks.append((c_name, nbr, nbr_proc, combined_query))

        if not tasks:
            return

        # Pass 2: dispatch all ask(phase='fuse') calls concurrently. Results
        # are collected in submission order (via ordered futures.result(),
        # NOT as_completed) so the original ordering is preserved.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    nbr_proc.ask,
                    query=combined_query,
                    phase='fuse',
                    **input_kwargs,
                )
                for (_c, _n, nbr_proc, combined_query) in tasks
            ]
            answer_chunks = [f.result() for f in futures]

        # Pass 3: apply side effects serially in the original order so that
        # fuse_history / detailed_log entries are identical to the sequential
        # implementation.
        for (c_name, nbr, _nbr_proc, combined_query), answer_chunk in zip(
            tasks, answer_chunks
        ):
            if answer_chunk is None:
                continue

            proc_map[c_name].add_fuse_history(
                combined_query, answer_chunk.gist, nbr
            )

            if self.detailed_log is not None:
                current_iteration = self.detailed_log['current_iteration']
                current_iteration['fuse_phase'].append(
                    {
                        'from_processor': c_name,
                        'to_processor': nbr,
                        'query': answer_chunk.executor_content or combined_query,
                        'answer': answer_chunk.gist,
                    }
                )

    # ------------------------------------------------------------------
    # go_down: broadcast + link_form
    # ------------------------------------------------------------------

    @logging_func_with_count
    def go_down(
        self, winning_chunk: Chunk, chunks: List[Chunk], **input_kwargs: Any
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
        *args: Any,
        **kwargs: Any,
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
        self._total_links_added = 0
        self.reset_usage_stats()
        answer = ''
        weight_score = 0.0

        max_iters = self.config.max_iter_num

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
            winning_chunk = self.uptree_competition(chunks)

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

            # Downtree + link_form
            self.go_down(winning_chunk, chunks, **input_params)

            # Fusion
            self.fuse_processor(
                chunks, query, winning_chunk=winning_chunk, **input_params
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
                'links_added': self._iter_links_added,
            }
            self.iteration_history.append(iteration_info)

            self.detailed_log['iterations'].append(
                self.detailed_log['current_iteration']
            )

        # Fallback (not normally reached)
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
