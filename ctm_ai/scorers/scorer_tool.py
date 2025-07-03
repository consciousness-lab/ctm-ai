from typing import Any, List

from openai import OpenAI

from ..messengers import Message
from ..utils import score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer('tool_scorer')
class ToolScorer(BaseScorer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_scorer()

    def init_scorer(self) -> None:
        self.scorer = OpenAI()

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message]) -> float:
        query = messages[-1].query
        gist = messages[-1].gist
        response = self.scorer.chat.completions.create(
            model='gpt-4o',
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
            print(f'[Directness GPT Response] {score_text}')
            try:
                score = float(score_text)
                scores.append(min(max(score / 10.0, 0.0), 1.0))
            except (ValueError, TypeError):
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0
