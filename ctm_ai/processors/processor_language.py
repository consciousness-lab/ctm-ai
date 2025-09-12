from typing import Any, Dict, List

from .processor_base import BaseProcessor


@BaseProcessor.register_processor('language_processor')
class LanguageProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

    def _init_info(self, *args: Any, **kwargs: Any) -> None:
        self.system_prompt = 'You are an expert in language understanding. Your task is to analyze the provided text and answer questions about it.'

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        return [{'role': 'user', 'content': f'Query: {query}\n'}]
