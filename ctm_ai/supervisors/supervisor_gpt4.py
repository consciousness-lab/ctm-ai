from typing import Any, Optional

from openai import OpenAI

from ..utils import (
    info_exponential_backoff,
    logprobs_to_softmax,
    score_exponential_backoff,
)
from .supervisor_base import BaseSupervisor


@BaseSupervisor.register_supervisor('gpt4_supervisor')
class GPT4Supervisor(BaseSupervisor):
    def init_supervisor(self) -> None:
        self.model = OpenAI()

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> Optional[str]:
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
        return responses.choices[0].message.content or None

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_score(self, query: str, gist: str, *args: Any, **kwargs: Any) -> float:
        if not gist:
            return 0.0

        response = self.model.chat.completions.create(
            model='gpt-4-0125-preview',
            messages=[
                {
                    'role': 'user',
                    'content': f'Is the information ({gist}) related to the query ({query})? Answer with "Yes" or "No".',
                },
            ],
            max_tokens=50,
            logprobs=True,
            top_logprobs=20,
        )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        logprob_dict = {logprob.token: logprob.logprob for logprob in top_logprobs}
        probs = logprobs_to_softmax(
            [logprob_dict.get('Yes', 0), logprob_dict.get('No', 0)]
        )
        return probs[0]
