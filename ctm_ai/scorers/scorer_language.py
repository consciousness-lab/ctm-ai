from typing import Any, List

from litellm import completion

from ..messengers import Message
from ..utils import logprobs_to_softmax, score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer('language_scorer')
class LanguageScorer(BaseScorer):
    def init_scorer(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the scorer using LiteLLM."""
        # Use default model for language scoring if not specified
        kwargs.setdefault('model', 'gpt-4o-mini')
        super().init_scorer(*args, **kwargs)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message]) -> float:
        """Ask relevance using LiteLLM with logprob analysis."""
        if not messages or not messages[-1].query or not messages[-1].gist:
            return 0.0

        query = messages[-1].query
        gist = messages[-1].gist

        try:
            response = completion(
                model=self.relevance_model,
                messages=[
                    {
                        'role': 'user',
                        'content': f"Is the information ({gist}) related with the query ({query})? Answer with 'Yes' or 'No'.",
                    }
                ],
                max_tokens=50,
                logprobs=True,
                top_logprobs=20,
                temperature=0.0,
            )

            if (
                response.choices
                and response.choices[0].logprobs
                and response.choices[0].logprobs.content
                and response.choices[0].logprobs.content[0]
                and response.choices[0].logprobs.content[0].top_logprobs
            ):
                top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                logprob_dict = {
                    logprob.token: logprob.logprob for logprob in top_logprobs
                }
                probs = logprobs_to_softmax(
                    [logprob_dict.get('Yes', 0), logprob_dict.get('No', 0)]
                )
                return probs[0]
            else:
                return 0.0

        except Exception as e:
            print(f'Error in language scorer relevance: {e}')
            # Fallback to parent class method
            return super().ask_relevance(messages)
