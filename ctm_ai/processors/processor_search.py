from typing import Any, Dict, List, Optional

import numpy as np
from google import genai
from google.genai import types
from litellm import completion

from ..chunks import Chunk
from ..scorers import BaseScorer
from .processor_base import BaseProcessor

SEARCH_PROMPT = """You should utilize the information in the context history and the current response from the search tool to generate a additional question about the query. Your additional question should be potentially answerable by other modality models and about specific information that you are not sure about. Your additional question should be just about what kind of information you need to get from other modality models or other tools like search engine, nothing else about the task or original query should be included. For example, what is the facial expression of the person, what text of the image, etc. The question needs to be short and clean.
There is the query: {query}
There is the response from the search tool: {response}
There is some additional information: {additional_information}
"""


@BaseProcessor.register_processor('search_processor')
class SearchProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

    def _generate_additional_info(
        self,
        query: str,
        response: str,
        **kwargs: Any,
    ) -> str:
        content = ''
        if len(self.fuse_history) > 0:
            content += '\nThere are extra information from other processors:\n'
            for i, item in enumerate(self.fuse_history, 1):
                content += f'{i}. {item["answer"]}\n'

        if len(self.winner_answer) > 0:
            content += '\nThere are some previous answers to the same query, think further based on this answer. These answers may not be correct, but it can provide some information.\n'
            for i, item in enumerate(self.winner_answer, 1):
                content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        additional_question = SEARCH_PROMPT.format(
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
        clean_query = query
        client = genai.Client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        system_instruction = f"""You are an expert search agent specializing in understanding the query. 
Search for answers about the query: {query}."""

        config = types.GenerateContentConfig(
            tools=[grounding_tool], system_instruction=system_instruction
        )
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=query,
            config=config,
        )
        executor_output = {'response': '', 'additional_question': ''}
        executor_output['response'] = response.text

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
        print(additional_content)
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
