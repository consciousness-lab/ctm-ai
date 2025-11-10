import re
from typing import Any, Dict, Optional, Tuple, Union

from ..utils import info_exponential_backoff, score_exponential_backoff
from .supervisor_base import BaseSupervisor


@BaseSupervisor.register_supervisor('language_supervisor')
class LanguageSupervisor(BaseSupervisor):
    def init_supervisor(self, *args: Any, **kwargs: Any) -> None:
        super().init_supervisor(*args, **kwargs)
        self.model_name = kwargs.get('supervisor_model', 'gemini/gemini-2.0-flash-lite')
        self.supervisors_prompt = kwargs.get('supervisors_prompt', '')
        self._original_context: Optional[str] = None

    def extract_action_and_remaining(self, text: str) -> tuple[str, str]:
        pattern = r'The specific action should be taken is:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)

        if match:
            action = match.group(1).strip()
            remaining_text = re.sub(pattern, '', text).strip()
            return remaining_text, action

        return text, ''

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> Optional[str]:
        answer, action = self.extract_action_and_remaining(context)
        return answer

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_score(self, query: str, gist: str, *args: Any, **kwargs: Any) -> str:
        if self._original_context:
            answer, action = self.extract_action_and_remaining(self._original_context)
            return action
        answer, action = self.extract_action_and_remaining(gist)
        return action

    def ask(
        self, query: str, context: str, action_history: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[str, None], Any]:
        self._original_context = context
        gist = self.ask_info(query, context)
        score = self.ask_score(query, gist)
        return gist, score
