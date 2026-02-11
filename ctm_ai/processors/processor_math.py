import os
from typing import Any, Dict, List, Optional

import requests
from litellm import completion

from ..chunks import Chunk
from .processor_base import BaseProcessor
from .utils import parse_json_response_with_scores

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

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """MathProcessor doesn't use this method as it has custom Wolfram API flow."""
        # This method is required by BaseProcessor but not used in custom ask()
        return None

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

        # Step 1: Call Wolfram|Alpha API to get computational result
        wolfram_response = self._call_wolfram_api(query)

        # If Wolfram API failed, skip this processor
        if wolfram_response.startswith('Error calling Wolfram|Alpha API:'):
            return None

        # Step 2: Generate structured output with answer, additional_questions, and scores
        structured_prompt = self._build_structured_prompt(clean_query, wolfram_response)

        executor_output = self._ask_for_structured_output(
            structured_prompt, *args, **kwargs
        )

        # Ensure we have the response from Wolfram result
        if not executor_output.get('response'):
            executor_output['response'] = wolfram_response

        # Add to history
        self.add_all_context_history(
            clean_query,
            executor_output['response'],
            executor_output.get('additional_questions', []),
        )

        # Extract scores using the base processor method
        scorer_output = self._extract_scores_from_executor_output(executor_output)
        additional_questions = executor_output.get('additional_questions', [])

        # Create chunk
        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_questions=additional_questions,
        )
        return chunk

    def _build_structured_prompt(self, query: str, wolfram_result: str) -> str:
        """Build prompt that asks for structured output with self-evaluation scores."""
        context_info = ''
        if len(self.fuse_history) > 0:
            context_info += '\nExtra information from other processors:\n'
            for i, item in enumerate(self.fuse_history, 1):
                context_info += f'{i}. {item["answer"]}\n'

        if len(self.winner_answer) > 0:
            context_info += (
                '\nPrevious answers to consider (may not be fully correct):\n'
            )
            for i, item in enumerate(self.winner_answer, 1):
                context_info += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        from .utils import JSON_FORMAT_SCORE

        prompt = f"""Query: {query}

Wolfram|Alpha Result:
{wolfram_result}
{context_info}

Based on the Wolfram|Alpha result above, please:
1. Provide a comprehensive response to the query, explaining the mathematical/computational result clearly
2. Generate an additional question if you need more information from other modality models (e.g., "what is shown in the diagram?", "what additional context is needed?"). If no additional information is needed, leave it empty.
3. Self-evaluate your response with relevance, confidence, and surprise scores.

{JSON_FORMAT_SCORE}"""

        return prompt

    def _ask_for_structured_output(
        self, prompt: str, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get structured output with self-evaluation scores using litellm."""
        call_kwargs = {
            **self._completion_kwargs,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': self.max_tokens,
            'n': self.return_num,
            'temperature': self.temperature,
            **kwargs,
        }
        response = completion(**call_kwargs)

        content = response.choices[0].message.content

        # Parse the JSON response with scores
        parsed = parse_json_response_with_scores(
            content, default_additional_questions=[]
        )

        return parsed
