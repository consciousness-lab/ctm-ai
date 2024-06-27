from typing import Any, Optional

from openai import OpenAI

from ..utils import info_exponential_backoff, score_exponential_backoff
from .supervisor_base import BaseSupervisor


@BaseSupervisor.register_supervisor('gpt4_supervisor')
class GPT4Supervisor(BaseSupervisor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_supervisor()

    def init_supervisor(self) -> None:
        self.model = OpenAI()

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> Any:
        responses = self.model.chat.completions.create(
            model='gpt-4-turbo-preview',
            messages=[
                {
                    'role': 'user',
                    'content': f'The following is detailed information on the topic: {context}. Based on this information, answer the question: {query}. Answer with a few words:',
                }
            ],
            max_tokens=300,
            n=1,
        )
        answer = (
            responses.choices[0].message.content
            if responses.choices[0].message.content
            else 'FAILED'
        )
        return answer

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_score(
        self,
        query: str,
        gist: str,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.model.chat.completions.create(
                    model='gpt-4-0125-preview',
                    messages=[
                        {
                            'role': 'user',
                            'content': f"How related is the information ({gist}) with the query ({query})? We want to make sure that the information includes a person's name as the answer. Answer with a number from 0 to 5 and do not add any other thing.",
                        },
                    ],
                    max_tokens=50,
                )
                score = (
                    float(response.choices[0].message.content.strip()) / 5
                    if response.choices[0].message.content
                    else 0.0
                )
                return score
            except Exception as e:
                print(f'Attempt {attempt + 1} failed: {e}')
                if attempt < max_attempts - 1:
                    print('Retrying...')
                else:
                    print('Max attempts reached. Returning default score.')
        return 0.0
