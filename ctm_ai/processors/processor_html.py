from typing import Any, Dict, List, Optional

from .processor_base import BaseProcessor
from .processor_webagent_base import WebAgentBaseProcessor
from .prompts.webagent_prompts import HTML_SYSTEM_PROMPT, build_html_user_prompt


@BaseProcessor.register_processor("html_processor")
class HTMLProcessor(WebAgentBaseProcessor):
    """Web-agent processor specialised in HTML source analysis."""

    def _build_web_prompt(
        self,
        objective: str,
        action_history: str,
        action_space: str,
        other_info: str,
        phase: str = "initial",
        **kwargs: Any,
    ) -> str:
        # WebConsciousTuringMachine routes html content via self._html;
        # fall back to the generic text kwarg for direct calls.
        html = getattr(self, "_html", None) or kwargs.get("text", "") or ""
        return build_html_user_prompt(
            objective=objective,
            html=html,
            action_history=action_history,
            action_space=action_space,
            other_info=other_info,
            phase=phase,
        )

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        system_prompt = self.system_prompt or HTML_SYSTEM_PROMPT
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
