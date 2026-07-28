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

# --------------------------------------------------------------------------
# Pluggable chunk-scoring for the confidence-estimator study. CTM_SCORE_METHOD
# in {self, judge, logprob} chooses how the up-tree competition weight is set;
# CTM_CHUNK_LOG (a dir) collects per-chunk (self / judge / logprob) values.
# --------------------------------------------------------------------------
_SCORE_CLIENT = None


def _score_client():
    global _SCORE_CLIENT
    if _SCORE_CLIENT is None:
        from openai import OpenAI

        _SCORE_CLIENT = OpenAI(
            base_url=os.getenv('VLLM_API_BASE', 'http://localhost:8001/v1'),
            api_key=os.getenv('VLLM_API_KEY', 'dummy'),
        )
    return _SCORE_CLIENT


_NOTHINK = {'chat_template_kwargs': {'enable_thinking': False}}


def _judge_score(query, gist):
    import re

    try:
        r = _score_client().chat.completions.create(
            model='Qwen3-8B',
            messages=[{'role': 'user', 'content': (
                f'Query: {query[:1500]}\nResponse: {gist[:2000]}\nEstimate the '
                'probability from 0.00 to 1.00 that this response CORRECTLY and '
                'COMPLETELY solves the query. Reply with ONLY the number.'
            )}],
            max_tokens=8, temperature=0.0, extra_body=_NOTHINK,
        )
        m = re.search(r'[01]?\.\d+|[01]', r.choices[0].message.content or '')
        if not m:
            return None
        v = float(m.group())
        return max(0.0, min(1.0, v / 100 if v > 1 else v))
    except Exception:  # noqa: BLE001
        return None


_SELF_DECOUPLED_PROMPT = (
    'You wrote the response below to the task. Evaluate ONLY this response.\n'
    'Task: {q}\nResponse: {a}\n\n'
    'Score each from 0.0 to 1.0:\n'
    '- relevance: does the response provide specific, actionable data that answers '
    'the task? If it says "I cannot"/"please provide" or gives guidance instead of '
    'actual data, relevance MUST be 0.0-0.2; only responses with ACTUAL DATA from '
    'tool calls deserve 0.6+.\n'
    '- confidence: how certain that the information is correct? If no tool was '
    'successfully called or a tool errored, confidence MUST be 0.0-0.3.\n'
    '- surprise: does it bring novel information beyond the task context?\n\n'
    'Reply with ONLY a JSON object: '
    '{{"relevance": <0-1>, "confidence": <0-1>, "surprise": <0-1>}}'
)


def _self_decoupled_score(query, gist):
    """CTM's own r/c/s self-scoring, but in a SEPARATE forward pass (decoupled
    from answer generation). Returns the CTM weight r + c + 0.2*s (0..2.2)."""
    import re

    try:
        r = _score_client().chat.completions.create(
            model='Qwen3-8B',
            messages=[{'role': 'user', 'content': _SELF_DECOUPLED_PROMPT.format(
                q=query[:1500], a=gist[:2000])}],
            max_tokens=120, temperature=0.0, extra_body=_NOTHINK,
        )
        txt = r.choices[0].message.content or ''

        def g(name):
            m = re.search(name + r'"?\s*:\s*([01]?\.\d+|[01](?:\.0+)?)', txt)
            return float(m.group(1)) if m else None

        rel, con, sur = g('relevance'), g('confidence'), g('surprise')
        if rel is None or con is None:
            return None
        return rel + con + 0.2 * (sur or 0.0)
    except Exception:  # noqa: BLE001
        return None


def _logprob_score(query, gist):
    import math

    try:
        r = _score_client().chat.completions.create(
            model='Qwen3-8B',
            messages=[{'role': 'user', 'content': (
                f'Query: {query[:1500]}\nResponse: {gist[:2000]}\nDoes the response '
                'correctly and completely solve the query? Answer with exactly one '
                'word: Yes or No.'
            )}],
            max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20,
            extra_body=_NOTHINK,
        )
        top = r.choices[0].logprobs.content[0].top_logprobs
        ly = ln = None
        for t in top:
            tk = t.token.strip().lower()
            if tk == 'yes' and ly is None:
                ly = t.logprob
            elif tk == 'no' and ln is None:
                ln = t.logprob
        if ly is None and ln is None:
            return None
        if ly is None:
            ly = ln - 10.0
        if ln is None:
            ln = ly - 10.0
        mx = max(ly, ln)
        ey, en = math.exp(ly - mx), math.exp(ln - mx)
        return ey / (ey + en)
    except Exception:  # noqa: BLE001
        return None


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

        # Per-instance override of the class-level link_form relevance
        # threshold. When the config specifies ``link_form_threshold``, use it;
        # otherwise fall back to the class default.
        cfg_lft = getattr(self.config, 'link_form_threshold', None)
        if cfg_lft is not None:
            self.LINK_FORM_THRESHOLD = float(cfg_lft)

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

                answer_chunk = nbr_proc.ask(
                    query=combined_query, phase='fuse', **input_kwargs
                )
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

    def _rescore_chunks(self, chunks, query, iteration=0):
        """Override each chunk's competition weight with the chosen estimator
        (CTM_SCORE_METHOD) and log per-chunk self/judge/logprob values (iter 0)."""
        # CTM-AI's canonical scoring is 'self_decoupled' (r+c+0.2s in a separate
        # forward pass). 'self' is the legacy coupled ablation.
        method = os.getenv('CTM_SCORE_METHOD', 'self_decoupled')
        log_dir = os.getenv('CTM_CHUNK_LOG')
        if method == 'self' and not log_dir:
            return
        snap = [(c, c.relevance, c.confidence, c.surprise, c.weight) for c in chunks]
        vals = {}
        fnmap = {'judge': _judge_score, 'logprob': _logprob_score,
                 'self_decoupled': _self_decoupled_score}
        if method in fnmap:
            fn = fnmap[method]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(8, len(chunks)))
            ) as ex:
                futs = {ex.submit(fn, query, c.gist or ''): idx
                        for idx, c in enumerate(chunks)}
                for f in concurrent.futures.as_completed(futs):
                    vals[futs[f]] = f.result()
        for idx, (c, sr, sc, ss, sw) in enumerate(snap):
            v = vals.get(idx)
            if v is not None:
                if method == 'self_decoupled':
                    c.weight = v  # already r + c + 0.2s on the 0..2.2 scale
                else:
                    c.weight, c.confidence = v * 2.2, v
            if log_dir and iteration == 0:
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    p = os.path.join(log_dir, f'{os.getpid()}.jsonl')
                    with open(p, 'a') as f:
                        f.write(json.dumps({
                            'query': query[:2000], 'processor': c.processor_name,
                            'gist': (c.gist or '')[:3000], 'self_relevance': sr,
                            'self_confidence': sc, 'self_surprise': ss,
                            'self_weight': sw,
                            'judge': v if method == 'judge' else None,
                            'logprob': v if method == 'logprob' else None,
                            'self_decoupled': v if method == 'self_decoupled' else None,
                        }, ensure_ascii=False) + '\n')
                except Exception:  # noqa: BLE001
                    pass

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
            self._rescore_chunks(chunks, query, i)
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
