import os
from typing import Any

import requests

from ..utils import info_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('search_engine_executor')
class SearchEngineExecutor(BaseExecutor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = os.environ['GOOGLE_API_KEY']
        self.cse_id = os.environ['GOOGLE_CSE_ID']
        self.url = 'https://www.googleapis.com/customsearch/v1'

    @info_exponential_backoff()
    def ask(self, messages: str, *args: Any, **kwargs: Any) -> str:
        params = {'key': self.api_key, 'cx': self.cse_id, 'q': messages}
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            search_results = response.json()
            content = ''
            for item in search_results.get('items', []):
                content += item.get('snippet', '') + '\n'
            return content
        except requests.exceptions.HTTPError as err:
            print(f'HTTP error occurred: {err}')
            return ''
        except Exception as err:
            print(f'Other error occurred: {err}')
            return ''
