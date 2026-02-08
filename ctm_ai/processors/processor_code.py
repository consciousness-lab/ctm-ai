from typing import Any, Dict, List

from .processor_base import BaseProcessor


@BaseProcessor.register_processor('code_processor')
class CodeProcessor(BaseProcessor):
    REQUIRED_KEYS = []

    def _init_info(self, *args: Any, **kwargs: Any) -> None:
        # Use system_prompt from config if provided, otherwise use default
        if not self.system_prompt:
            self.system_prompt = 'You are an expert in code writing.'

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        return [{'role': 'user', 'content': query}]
