import os
from typing import Any, Optional

import requests
from litellm import completion

from ..chunks import Chunk
from ..scorers import BaseScorer
from .processor_base import BaseProcessor

MATH_PROMPT = """You should utilize the information in the context history and the current response from the Wolfram|Alpha tool to generate an additional question about the query. Your additional question should be potentially answerable by other modality models and about specific information that you are not sure about. Your additional question should be just about what kind of information you need to get from other modality models or other tools, nothing else about the task or original query should be included. For example, what additional context is needed, what related concepts should be explored, etc. The question needs to be short and clean.
There is the query: {query}
There is the response from Wolfram|Alpha: {response}
There is some additional information: {additional_information}
"""

WOLFRAM_API_BASE_URL = 'https://www.wolframalpha.com/api/v1/llm-api'


@BaseProcessor.register_processor('math_processor')
class MathProcessor(BaseProcessor):
    REQUIRED_KEYS = ['WOLFRAM_APPID']

    def __init__(
        self, name: str, group_name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(name, group_name, *args, **kwargs)
        self.wolfram_appid = os.environ.get('WOLFRAM_APPID', '')
        self.max_chars = kwargs.get('max_chars', 6800)

    def _call_wolfram_api(
        self,
        query: str,
        max_chars: Optional[int] = None,
    ) -> str:
        """Call Wolfram|Alpha LLM API and return the response text."""
        params = {
            'input': query,
            'appid': self.wolfram_appid,
            'maxchars': max_chars or self.max_chars,
        }

        try:
            response = requests.get(
                WOLFRAM_API_BASE_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            return f'Error calling Wolfram|Alpha API: {str(e)}'

    def _generate_additional_info(
        self,
        query: str,
        response: str,
        **kwargs: Any,
    ) -> str:
        """Generate additional context for creating follow-up questions."""
        content = ''
        if len(self.fuse_history) > 0:
            content += '\nThere are extra information from other processors:\n'
            for i, item in enumerate(self.fuse_history, 1):
                content += f'{i}. {item["answer"]}\n'

        if len(self.winner_answer) > 0:
            content += '\nThere are some previous answers to the same query, think further based on this answer. These answers may not be correct, but it can provide some information.\n'
            for i, item in enumerate(self.winner_answer, 1):
                content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        additional_question = MATH_PROMPT.format(
            query=query, response=response, additional_information=content
        )
        return additional_question

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        is_fuse: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        """Process mathematical/computational queries using Wolfram|Alpha LLM API."""
        clean_query = query

        wolfram_response = self._call_wolfram_api(query)

        # If Wolfram API failed, skip this processor
        if wolfram_response.startswith('Error calling Wolfram|Alpha API:'):
            return None

        executor_output = {'response': '', 'additional_question': ''}
        executor_output['response'] = wolfram_response

        additional_content = self._generate_additional_info(
            query=clean_query, response=executor_output['response']
        )
        response = completion(
            model=self.model,
            messages=[{'role': 'user', 'content': additional_content}],
            max_tokens=self.max_tokens,
            n=self.return_num,
            *args,
            **kwargs,
        )
        executor_output['additional_question'] = response.choices[0].message.content

        self.add_all_context_history(
            clean_query,
            executor_output['response'],
            executor_output['additional_question'],
        )

        scorer = BaseScorer(*args, **kwargs)
        scorer_output = scorer.ask(query=clean_query, messages=executor_output)
        additional_question = executor_output['additional_question'] or ''

        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )
        return chunk
