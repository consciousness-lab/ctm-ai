"""
ToolProcessor for ToolBench – handles tool/API calling in CTM framework.

Two-stage pipeline (applies to every phase):
  Stage 1 – Tool decision + execution   (with exponential backoff)
  Stage 2 – Answer synthesis + scoring   (via ask_executor, also with backoff)
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from litellm import completion
from numpy.typing import NDArray

from ..chunks import Chunk
from ..utils import message_exponential_backoff
from .processor_base import BaseProcessor
from .prompts.tool_prompts import (
    DEFAULT_NUM_ADDITIONAL_QUESTIONS,
    TOOLBENCH_SYSTEM_PROMPT,
    TOOLBENCH_TOOL_DECISION_PROMPT,
    build_tool_stage2_prompt,
)


def clean_tools_for_vertex_ai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove fields not supported by Vertex AI from tool definitions."""
    cleaned_tools = []
    for tool in tools:
        if tool.get('type') == 'function' and 'function' in tool:
            function_def = tool['function'].copy()

            if 'parameters' in function_def:
                params = function_def['parameters'].copy()
                if 'optional' in params:
                    del params['optional']
                if 'properties' in params:
                    cleaned_properties = {}
                    for prop_name, prop_def in params['properties'].items():
                        cleaned_prop = prop_def.copy()
                        if 'example_value' in cleaned_prop:
                            del cleaned_prop['example_value']
                        cleaned_properties[prop_name] = cleaned_prop
                    params['properties'] = cleaned_properties
                function_def['parameters'] = params

            cleaned_tools.append({'type': 'function', 'function': function_def})
        else:
            cleaned_tools.append(tool)
    return cleaned_tools


def register_tool_processors(openai_function_names: List[str]) -> None:
    """Register tool processors dynamically based on available function names."""
    for name in openai_function_names:
        BaseProcessor._processor_registry[name] = ToolProcessor


@BaseProcessor.register_processor('tool_processor')
class ToolProcessor(BaseProcessor):
    """
    Processor for ToolBench that uses a two-stage pipeline for all phases:

      Stage 1 – LLM decides whether to call a tool; if yes, executes it.
      Stage 2 – LLM synthesizes a final answer (+ scores, per phase) based on
                the complete Stage-1 outcome (tool called or not, result, etc.).

    This ensures scores are generated *after* seeing the full picture and in the
    *same* call as the final answer, consistent with BaseProcessor's approach.
    """

    TOOL_EXEC_RETRIES = 3

    def __init__(
        self, name: str, group_name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(name, group_name, *args, **kwargs)
        self.api_manager = kwargs.get('api_manager')
        self.system_prompt = kwargs.get('system_prompt') or TOOLBENCH_SYSTEM_PROMPT
        self.num_additional_questions = kwargs.get(
            'num_additional_questions', DEFAULT_NUM_ADDITIONAL_QUESTIONS
        )

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------

    def _build_executor_content(
        self,
        query: str,
        phase: str = 'initial',
        **kwargs: Any,
    ) -> str:
        """Build Stage 1 content: task + context + tool-decision protocol.

        Context (fuse_history / winner_answer) is only included for initial and
        link_form phases.  The fuse phase does not need it.

        The context is placed *between* the task description and the DECISION
        section so the LLM sees the full picture before deciding whether to
        call the tool.
        """
        context_parts: List[str] = []

        if phase in ('initial', 'link_form'):
            if self.fuse_history:
                context_parts.append(
                    'The following extra information was obtained from other tools:'
                )
                for i, item in enumerate(self.fuse_history, 1):
                    context_parts.append(
                        f'{i}. {item["processor_name"]}: {item["answer"]}'
                    )

            if self.winner_answer:
                context_parts.append(
                    'The following are previous answers to the same query:'
                )
                for i, item in enumerate(self.winner_answer, 1):
                    context_parts.append(
                        f'{i}. {item["processor_name"]}: {item["answer"]}'
                    )

        context_str = '\n'.join(context_parts) if context_parts else ''

        return TOOLBENCH_TOOL_DECISION_PROMPT.format(
            query=query, function_name=self.name, context=context_str
        )

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Build messages for Stage 1 LLM call."""
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': query},
        ]

    @message_exponential_backoff()
    def _ask_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Any:
        """Stage 1 LLM call with tool definitions and exponential backoff."""
        call_kwargs = {
            **self._completion_kwargs,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'tools': tools,
            'tool_choice': 'auto',
        }
        return completion(**call_kwargs)

    def _tool_decision_and_execute(
        self,
        query: str,
        api_manager: Any,
        function_name: str,
    ) -> Tuple[bool, Optional[str], Optional[str], str]:
        """
        Stage 1: decide whether to call a tool and execute if needed.

        Returns:
            (tool_called, tool_name, tool_args, raw_result)
        """
        if api_manager is None:
            return False, None, None, f'No API manager available for {function_name}'

        raw_tools = api_manager.funcs_to_all_info.get(function_name, [])
        if isinstance(raw_tools, dict):
            raw_tools = [{'type': 'function', 'function': raw_tools}]
        cleaned_tools = clean_tools_for_vertex_ai(raw_tools)

        messages = self.build_executor_messages(query)

        try:
            response = self._ask_with_tools(messages, cleaned_tools)
        except Exception as e:
            return False, None, None, f'Error calling LLM: {e}'

        if response is None:
            return False, None, None, 'LLM returned None after retries'

        msg = response.choices[0].message

        # Case 1: direct text answer (no tool call)
        if msg.content is not None and msg.tool_calls is None:
            return False, None, None, msg.content

        # Case 2: model chose to call a tool
        if msg.tool_calls is not None:
            tool_call = msg.tool_calls[0]
            func_name = getattr(tool_call.function, 'name', None) or function_name
            func_args = getattr(tool_call.function, 'arguments', '{}')

            if isinstance(func_args, dict):
                func_args_str = json.dumps(func_args, ensure_ascii=False)
            else:
                func_args_str = str(func_args) if func_args is not None else '{}'

            tool_result = None
            for attempt in range(self.TOOL_EXEC_RETRIES):
                try:
                    tool_result, status_code = api_manager.step(
                        action=func_name, input_str=func_args_str
                    )
                    if status_code in (0, 3):
                        break
                except Exception as e:
                    if attempt < self.TOOL_EXEC_RETRIES - 1:
                        continue
                    tool_result = json.dumps(
                        {
                            'error': f'tool execution failed: {type(e).__name__}: {e}',
                            'response': '',
                        }
                    )

            if tool_result is not None and not isinstance(tool_result, str):
                tool_result = str(tool_result)

            return True, func_name, func_args_str, tool_result or ''

        # Case 3: unexpected (no content, no tool_calls)
        return False, None, None, 'No valid response received from the model'

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ask(
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
        phase: str = 'initial',
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        """
        Two-stage pipeline (all phases):
          Stage 1 – Tool decision + optional execution (with backoff)
          Stage 2 – Synthesis + self-evaluation    (via ask_executor with backoff)
        """
        if api_manager is None:
            api_manager = self.api_manager

        # ── Stage 1: Tool decision + execution ──
        executor_content = self._build_executor_content(query=query, phase=phase)

        from ..utils import logger

        logger.info(
            f'\n{self.name} received query (phase={phase}):\n{executor_content[:500]}...'
            if len(executor_content) > 500
            else f'\n{self.name} received query (phase={phase}):\n{executor_content}'
        )

        tool_called, tool_name, tool_args, raw_result = self._tool_decision_and_execute(
            query=executor_content,
            api_manager=api_manager,
            function_name=self.name,
        )

        # ── Stage 2: Synthesis + scoring ──
        # Context (fuse_history / winner_answer) is passed for initial and
        # link_form so the LLM can synthesize a comprehensive answer.
        # The fuse phase only needs to produce a short answer—no context.
        include_context = phase in ('initial', 'link_form')
        stage2_prompt = build_tool_stage2_prompt(
            query=query,
            tool_called=tool_called,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_result=raw_result,
            phase=phase,
            fuse_history=self.fuse_history if include_context else None,
            winner_answer=self.winner_answer if include_context else None,
            num_additional_questions=(
                self.num_additional_questions if phase == 'initial' else 0
            ),
        )

        default_qs = (
            ['Can you provide more information?']
            if phase == 'initial' and self.num_additional_questions > 0
            else []
        )

        stage2_output = self.ask_executor(
            messages=[{'role': 'user', 'content': stage2_prompt}],
            default_additional_questions=default_qs,
        )

        # ── Build Chunk based on phase ──
        if stage2_output is None:
            return None

        if phase == 'fuse':
            response_str = stage2_output.get('response', '')
            self.add_fuse_history(query, response_str, self.name)
            return Chunk(
                time_step=0,
                processor_name=self.name,
                gist=response_str,
                relevance=0.0,
                confidence=0.0,
                surprise=0.0,
                weight=0.0,
                additional_questions=[],
                executor_content=executor_content,
            )

        if phase == 'link_form':
            response_str = stage2_output.get('response', '')
            relevance = float(stage2_output.get('relevance', 0.5))
            return Chunk(
                time_step=0,
                processor_name=self.name,
                gist=response_str,
                relevance=relevance,
                confidence=0.0,
                surprise=0.0,
                weight=relevance,
                additional_questions=[],
                executor_content=executor_content,
            )

        # ── Initial phase: full processing ──
        if stage2_output.get('response') is None:
            return None

        self.add_all_context_history(
            query,
            stage2_output['response'],
            stage2_output.get('additional_questions', []),
        )

        scorer_output = self._extract_scores_from_executor_output(stage2_output)
        additional_questions = stage2_output.get('additional_questions', [])

        return self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=stage2_output,
            additional_questions=additional_questions,
            executor_content=executor_content,
        )
