from typing import Any, Dict, List

from .processor_base import BaseProcessor


@BaseProcessor.register_processor('language_processor')
class LanguageProcessor(BaseProcessor):
    REQUIRED_KEYS = []

    def _init_info(self, *args: Any, **kwargs: Any) -> None:
        # Use system_prompt from config if provided, otherwise use default
        if not self.system_prompt:
            self.system_prompt = 'You are an expert in language understanding. Your task is to analyze the provided text and answer questions about it.'

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._init_info(*args, **kwargs)
        text = kwargs.get('text', ' ')
        if text:
            language_message = {
                'role': 'user',
                'content': f'{query}\n The relevant text of the query is: {text}\n',
            }
        else:
            language_message = {
                'role': 'user',
                'content': f'{query}\n',
            }

        all_messages = [{'role': 'system', 'content': self.system_prompt}]
        all_messages.append(language_message)
        return all_messages
