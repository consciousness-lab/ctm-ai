from typing import Any, Dict, List, Optional

import numpy as np
from litellm import completion

from ..chunks import Chunk
from ..utils import message_exponential_backoff
from .processor_base import BaseProcessor
from .prompts.webagent_prompts import parse_webagent_response


class WebAgentBaseProcessor(BaseProcessor):
    """Shared logic for all web-agent processors (not registered)."""

    # ------------------------------------------------------------------
    # Public entry point — captures web-specific kwargs
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio=None,
        audio_path: Optional[str] = None,
        video_frames=None,
        video_frames_path=None,
        video_path: Optional[str] = None,
        api_manager: Any = None,
        phase: str = "initial",
        *args: Any,
        **kwargs: Any,
    ) -> Optional[Chunk]:
        # Web-agent context — consumed by _build_executor_content and
        # build_executor_messages via instance attributes.
        self._action_history: str = kwargs.get("action_history", "No previous actions")
        self._action_space: str = kwargs.get("action_space", "Standard browser actions")

        # Per-modality inputs routed by WebConsciousTuringMachine.
        # Each processor's _build_web_prompt / build_executor_messages uses
        # the attribute that corresponds to its own modality.
        self._axtree: str = kwargs.get("axtree", text or "")
        self._html: str = kwargs.get("html", text or "")
        self._screenshot_b64: Optional[str] = kwargs.get("screenshot")

        # External contextual info (e.g. open tabs) injected by the agent.
        self._other_info_extra: str = kwargs.get("other_info", "")

        return super().ask(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
            api_manager=api_manager,
            phase=phase,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Prompt construction — delegates to subclass
    # ------------------------------------------------------------------

    def _build_executor_content(
        self,
        query: str,
        phase: str = "initial",
        **kwargs: Any,
    ) -> str:
        """Build the fully-formatted web-agent prompt.

        Assembles ``other_info`` from three sources (in priority order):
          1. External context injected by the agent (``_other_info_extra``,
             e.g. open-tab list).
          2. Answers received from sibling processors via fuse (fuse_history).
          3. Previous winning answers from earlier CTM iterations (winner_answer).
        """
        other_info_parts: List[str] = []

        # 1. External context (always first so the model sees it prominently)
        extra = getattr(self, "_other_info_extra", "")
        if extra:
            other_info_parts.append(extra)

        # 2. Fuse history — answers from sibling processors
        for item in self.fuse_history:
            other_info_parts.append(f'[{item["processor_name"]}]: {item["answer"]}')

        # 3. Previous CTM round winners
        if self.winner_answer and phase == "initial":
            other_info_parts.append(
                "There are previous answers for "
                "the same step. Think further based on this:"
            )
            for item in self.winner_answer:
                other_info_parts.append(
                    f'[{item["processor_name"]} – previous round]: {item["answer"]}'
                )

        other_info = "\n".join(other_info_parts) if other_info_parts else "None"

        return self._build_web_prompt(
            objective=query,
            action_history=getattr(self, "_action_history", "No previous actions"),
            action_space=getattr(self, "_action_space", "Standard browser actions"),
            other_info=other_info,
            phase=phase,
            **kwargs,
        )

    def _build_web_prompt(
        self,
        objective: str,
        action_history: str,
        action_space: str,
        other_info: str,
        phase: str = "initial",
        **kwargs: Any,
    ) -> str:
        """Subclasses must implement this to produce the final prompt string."""
        raise NotImplementedError("Subclasses must implement _build_web_prompt")

    # ------------------------------------------------------------------
    # LLM call — custom parser extracts the ``action`` field
    # ------------------------------------------------------------------

    @message_exponential_backoff()
    def ask_executor(
        self,
        messages: List[Dict[str, Any]],
        default_additional_questions: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = {
            **self._completion_kwargs,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "n": self.return_num,
            **kwargs,
        }
        response = completion(**call_kwargs)
        content = response.choices[0].message.content
        return parse_webagent_response(content, default_additional_questions)

    # ------------------------------------------------------------------
    # Memory — store reasoning + action so next iteration has full context
    # ------------------------------------------------------------------

    def update(self, chunk: Chunk) -> None:
        """Override to store both reasoning and action in winner_answer."""
        if chunk.processor_name == self.name:
            return

        reasoning = self._extract_chunk_reasoning(chunk)
        action = chunk.gist

        parts = []
        if reasoning:
            parts.append(f"Reasoning: {reasoning}")
        if action:
            parts.append(f"Action: {action}")
        answer = "\n".join(parts) if parts else action

        self.winner_answer.append(
            {"processor_name": chunk.processor_name, "answer": answer}
        )

    @staticmethod
    def _extract_chunk_reasoning(chunk: Chunk) -> str:
        content = chunk.executor_content or ""
        prefix = "[Reasoning]: "
        if content.startswith(prefix):
            end = content.find("\n\n")
            return content[len(prefix) : end] if end != -1 else content[len(prefix) :]
        return ""

    def merge_outputs_into_chunk(
        self,
        name: str,
        executor_output: Dict[str, Any],
        scorer_output: Dict[str, float],
        additional_questions: Optional[List[str]] = None,
        executor_content: str = "",
    ) -> Chunk:
        # ``response`` in executor_output already holds the action string
        # (mapped by parse_webagent_response); ``reasoning`` holds the CoT text.
        action = executor_output.get("response", "")
        reasoning = executor_output.get("reasoning", "")

        enriched_content = executor_content
        if reasoning:
            enriched_content = f"[Reasoning]: {reasoning}\n\n{executor_content}"

        return Chunk(
            time_step=0,
            processor_name=name,
            gist=action,
            relevance=scorer_output["relevance"],
            confidence=scorer_output["confidence"],
            surprise=scorer_output["surprise"],
            weight=scorer_output["weight"],
            additional_questions=additional_questions or [],
            executor_content=enriched_content,
        )
