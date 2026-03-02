import concurrent.futures
from typing import Any, List, Optional, Tuple

from ..chunks import Chunk
from ..utils import logger, logging_func_with_count
from .ctm import ConsciousTuringMachine


class WebConsciousTuringMachine(ConsciousTuringMachine):
    """CTM specialised for web-agent tasks.

    Usage::

        ctm = WebConsciousTuringMachine(ctm_name='web')
        reasoning, action = ctm(
            query='Find the cheapest red shoes',
            axtree=axtree_str,
            html=html_str,
            screenshot=screenshot_b64,
            action_history='click("42")',
            action_space=action_space_str,
            other_info='Tab 0 (active): https://example.com',
        )
    """

    def __call__(
        self,
        query: str,
        *,
        axtree: str = '',
        html: str = '',
        screenshot: Optional[str] = None,
        action_history: str = '',
        action_space: str = '',
        other_info: str = '',
        force_final: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, str, str]:
        """Run one CTM step and return ``(reasoning, action, parsed_action)``.

        ``action``        – raw browser command string (e.g. ``click("42")``).
        ``reasoning``     – the winning processor's chain-of-thought text.
        ``parsed_action`` – action cleaned up by ``parse_answer``.
        """
        return self.forward(
            query=query,
            axtree=axtree,
            html=html,
            screenshot=screenshot,
            action_history=action_history,
            action_space=action_space,
            other_info=other_info,
            force_final=force_final,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Forward loop
    # ------------------------------------------------------------------

    def forward(
        self,
        query: str,
        axtree: str = '',
        html: str = '',
        screenshot: Optional[str] = None,
        action_history: str = '',
        action_space: str = '',
        other_info: str = '',
        force_final: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, str, str]:
        """Iterative CTM loop adapted for web agents.

        Returns ``(reasoning, action, parsed_action)``.
        """
        web_params: dict = {
            'axtree': axtree,
            'html': html,
            'screenshot': screenshot,
            'action_history': action_history,
            'action_space': action_space,
            'other_info': other_info,
        }

        step_log: dict = {'query': query, 'iterations': []}

        action = ''
        reasoning = ''

        for i in range(self.config.max_iter_num):
            iteration_log: dict = {
                'iteration': i,
                'all_chunks': [],
                'winning_chunk': None,
                'link_form_phase': [],
                'fuse_phase': [],
            }
            self.detailed_log = {'current_iteration': iteration_log}

            chunks = self._ask_web_processors(query, phase='initial', **web_params)

            for chunk in chunks:
                iteration_log['all_chunks'].append({
                    'processor_name': chunk.processor_name,
                    'gist': chunk.gist,
                    'executor_content': chunk.executor_content,
                    'relevance': chunk.relevance,
                    'confidence': chunk.confidence,
                    'surprise': chunk.surprise,
                    'weight': chunk.weight,
                    'additional_questions': chunk.additional_questions,
                })

            if not chunks:
                logger.warning('WebCTM: no processor returned a valid chunk.')
                step_log['iterations'].append(iteration_log)
                break

            winning_chunk = self.uptree_competition(chunks)
            action = winning_chunk.gist
            reasoning = self._extract_reasoning(winning_chunk)

            iteration_log['winning_chunk'] = {
                'processor_name': winning_chunk.processor_name,
                'gist': winning_chunk.gist,
                'executor_content': winning_chunk.executor_content,
                'reasoning': reasoning,
                'weight': winning_chunk.weight,
            }

            if (
                i == self.config.max_iter_num - 1
                or winning_chunk.weight >= self.config.output_threshold
            ):
                step_log['iterations'].append(iteration_log)
                break

            self.downtree_broadcast(winning_chunk)
            self.link_form(chunks, winning_chunk, **web_params)
            self.fuse_processor(chunks, query, **web_params)

            step_log['iterations'].append(iteration_log)

        self.detailed_log = None

        parsed_answer = self.parse_answer(
            answer=action,
            query=query,
            reasoning=reasoning,
            action_history=web_params.get('action_history', ''),
            force_final=force_final,
        )

        step_log['final_action'] = action
        step_log['final_reasoning'] = reasoning
        step_log['parsed_answer'] = parsed_answer
        step_log['force_final'] = force_final
        self.last_step_log = step_log

        return reasoning, action, parsed_answer

    # ------------------------------------------------------------------
    # Web-specific processor dispatch
    # ------------------------------------------------------------------

    @logging_func_with_count
    def ask_processors(
        self,
        query: str,
        text: Optional[str] = None,
        image=None,
        image_path: Optional[str] = None,
        audio=None,
        audio_path: Optional[str] = None,
        video_frames=None,
        video_frames_path=None,
        video_path: Optional[str] = None,
        api_manager: Any = None,
        phase: str = 'initial',
        **kwargs: Any,
    ) -> List[Chunk]:
        """Override base ask_processors to use web routing.

        When called from ``link_form`` or ``fuse_processor`` (which pass
        web_params as **kwargs), this method correctly dispatches each
        processor with its own modality data.
        """
        return self._ask_web_processors(query, phase=phase, **kwargs)

    def _ask_web_processors(
        self,
        query: str,
        phase: str = 'initial',
        **web_kwargs: Any,
    ) -> List[Chunk]:
        """Dispatch processors in parallel, routing each to its modality."""
        axtree = web_kwargs.get('axtree', web_kwargs.get('text', ''))
        html = web_kwargs.get('html', web_kwargs.get('text', ''))
        screenshot = web_kwargs.get('screenshot')

        shared = {
            'action_history': web_kwargs.get('action_history', ''),
            'action_space': web_kwargs.get('action_space', ''),
            'other_info': web_kwargs.get('other_info', ''),
            'phase': phase,
        }

        # Per-processor modality routing
        _modality: dict = {
            'axtree_processor': {'text': axtree},
            'html_processor': {'text': html},
            'screenshot_processor': {'screenshot': screenshot},
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    proc.ask,
                    query,
                    **_modality.get(proc.name, {}),
                    **shared,
                ): proc.name
                for proc in self.processor_graph.nodes
            }
            chunks = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk = future.result()
                    if chunk is not None:
                        chunks.append(chunk)
                except Exception as exc:
                    proc_name = futures[future]
                    logger.warning(f'WebCTM: processor {proc_name} raised: {exc}')

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reasoning(chunk: Chunk) -> str:
        """Pull the CoT reasoning out of executor_content if present."""
        content = chunk.executor_content or ''
        prefix = '[Reasoning]: '
        if content.startswith(prefix):
            end = content.find('\n\n')
            return content[len(prefix) : end] if end != -1 else content[len(prefix) :]
        return chunk.gist
