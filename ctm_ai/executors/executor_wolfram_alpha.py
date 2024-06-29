import os
from typing import Any, List, Union

import requests

from ..utils import logger, multi_info_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('wolfram_alpha_executor')
class WolframAlphaExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = os.environ.get('WOLFRAM_API_KEY')
        self.url = 'http://api.wolframalpha.com/v2/query'

    @multi_info_exponential_backoff()
    def ask(self, messages: str, *args: Any, **kwargs: Any) -> List[Union[str, None]]:
        params = {'input': messages, 'appid': self.api_key, 'output': 'json'}
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            search_results = response.json()
            content = ''
            for pod in search_results.get('queryresult', {}).get('pods', []):
                for subpod in pod.get('subpods', []):
                    content += subpod.get('plaintext', '') + '\n'
            return [content]
        except requests.exceptions.HTTPError as err:
            logger.error(f'HTTP error occurred: {err}')
            return []
        except Exception as err:
            logger.error(f'An error occurred: {err}')
            return []
