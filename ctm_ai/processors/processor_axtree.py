from typing import Any, Dict, List

from .processor_base import BaseProcessor
from .processor_webagent_base import WebAgentBaseProcessor
from .prompts.webagent_prompts import AXTREE_SYSTEM_PROMPT, build_axtree_user_prompt


@BaseProcessor.register_processor('axtree_processor')
class AXTreeProcessor(WebAgentBaseProcessor):
    """Web-agent processor specialised in accessibility-tree interpretation."""

    def _build_web_prompt(
        self,
        objective: str,
        action_history: str,
        action_space: str,
        other_info: str,
        phase: str = 'initial',
        **kwargs: Any,
    ) -> str:
        # WebConsciousTuringMachine routes axtree content via self._axtree;
        # fall back to the generic text kwarg for direct calls.
        axtree = getattr(self, '_axtree', None) or kwargs.get('text', '') or ''
        return build_axtree_user_prompt(
            objective=objective,
            axtree=axtree,
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
        system_prompt = self.system_prompt or AXTREE_SYSTEM_PROMPT
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query},
        ]
