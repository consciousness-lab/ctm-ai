from typing import Any, List

from litellm import completion

from ..messengers import Message
from ..utils import score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer('tool_scorer')
class ToolScorer(BaseScorer):
    def init_scorer(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the scorer using LiteLLM."""
        # Use default model for tool scoring if not specified
        kwargs.setdefault('model', 'gpt-4o')
        super().init_scorer(*args, **kwargs)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message]) -> float:
        """Ask relevance using LiteLLM for tool-based scoring."""
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
                        'content': (
                            f'Question: {query}\n'
                            f'Answer: {gist}\n\n'
                            'Evaluate how relevant does the answer respond to the question?.\n'
                            'Give a score from 0 to 10 based on the following criteria:\n'
                            '- 10 = Fully answers the question and do not need other information.\n'
                            '- 7 = Partially answers the question, but vague or lacks detail, or need more information.\n'
                            '- 3 = Does not answer the question, but its topic is somehow revelant to the question.\n'
                            '- 0 = The answer completely irrelevant to the question.\n\n'
                            'Respond with a single number only. Do not explain.'
                        ),
                    }
                ],
                max_tokens=10,
                n=5,
                temperature=0.7,
            )
            
            scores = []
            for choice in response.choices:
                score_text = choice.message.content.strip()
                print(f'[Tool Scorer LLM Response] {score_text}')
                try:
                    score = float(score_text)
                    scores.append(min(max(score / 10.0, 0.0), 1.0))
                except (ValueError, TypeError):
                    scores.append(0.0)

            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            print(f"Error in tool scorer relevance: {e}")
            # Fallback to parent class method
            return super().ask_relevance(messages)
