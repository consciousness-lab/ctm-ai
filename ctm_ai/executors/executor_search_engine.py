import os
from typing import Any, List

import requests

from ..messengers import Message
from ..utils import logger, message_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('search_engine_executor')
class SearchEngineExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = os.environ['GOOGLE_API_KEY']
        self.cse_id = os.environ['GOOGLE_CSE_ID']
        self.url = 'https://www.googleapis.com/customsearch/v1'

    @message_exponential_backoff()
    def ask(self, messages: List[Message], *args: Any, **kwargs: Any) -> Message:
        query = messages[-1].content
        params = {'key': self.api_key, 'cx': self.cse_id, 'q': query}
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            search_results = response.json()
            content = ''
            for item in search_results.get('items', []):
                content += item.get('snippet', '') + '\n'
            return Message(role='assistant', content=content, gist=content)
        except requests.exceptions.HTTPError as err:
            logger.error(f'HTTP error occurred: {err}')
            return Message(role='assistant', content='')
        except Exception as err:
            logger.error(f'Other error occurred: {err}')
            return Message(role='assistant', content='')
