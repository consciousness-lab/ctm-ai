from typing import Any, Dict

import numpy as np
from litellm import completion, embedding

from ..utils import configure_litellm, score_exponential_backoff
from .scorer_base import BaseScorer

RELEVANCE_PROMPT_OPTION = """
You are scoring a ONE-STEP web agent proposal for its task fit.

Inputs:
- Objective (the user's goal): {objective}
- Action history (most recent step, may be empty): {action_history}
- Candidate response and candidate action: {gist}
- Action space: {action_space}

Task: Give a relevance score in [0.0, 1.0] for how well this ONE step (response+action) DIRECTLY advances the Objective, considering the current progress implied by Action history.

Guidelines (think about both *what* and *where*):
- 1.0 Perfect: the action is exactly the next logical step toward the Objective (e.g., type into the correct field, click the correct submit), with no obvious detours.
- 0.8 High: clearly advances the goal but may miss minor specifics (e.g., target a general search instead of a scoped search).
- 0.6 Moderate: loosely on track but may be premature, redundant, or slightly mismatched with progress so far.
- 0.4 Low: weak connection; likely not the next useful step (e.g., clicking a non-essential UI element).
- 0.2 Barely: tenuous relation, mostly off-target.
- 0.0 None: unrelated to the goal or contradicts progress.

ONLY output a single number in [0.0, 1.0], e.g., 0.85.
"""

CONFIDENCT_PROMPT_OPTION = """
You are scoring how EXECUTABLE the proposed ONE-STEP web action is, given only its text form and minimal context.

Inputs:
- Objective: {objective}
- Action history (most recent step, may be empty): {action_history}
- Candidate response and candidate action: {gist}
- Action space: {action_space}

Definition:
Confidence = your estimate in [0.0, 1.0] that executing this action NOW will succeed (the target exists, is the right UI affordance, is likely interactable, and the step is correctly specified and not premature).

Heuristics you may use (text-only reasoning):
- Action is syntactically valid and unambiguous (parsable call signature, correct arg count).
- Verb↔UI affordance match (click→button/link/tab; type/input→textbox/field; select→dropdown).
- Target specificity looks sufficient (e.g., concrete id/selector/index), not vague.
- Step timing makes sense after Last action (e.g., type before submit).
- No obvious contradictions with Objective.

Scoring guide:
- 1.0 Very likely to execute successfully now.
- 0.8 Likely; minor uncertainty.
- 0.6 Uncertain; may fail depending on page specifics.
- 0.4 Unlikely; underspecified or mistimed.
- 0.2 Very unlikely; poorly specified.
- 0.0 Impossible; not a valid/parsable action.

ONLY output a single number in [0.0, 1.0], e.g., 0.70.
"""


class WebScorer(BaseScorer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_scorer(*args, **kwargs)

    def init_scorer(self, *args: Any, **kwargs: Any) -> None:
        self.model_name = kwargs.get('model', 'gemini/gemini-2.0-flash-lite')
        self.embedding_model = kwargs.get('embedding_model', 'text-embedding-3-small')
        self.relevance_model = kwargs.get('relevance_model', self.model_name)
        self.confidence_model = kwargs.get('confidence_model', self.model_name)
        self.surprise_model = kwargs.get('scorer_model', self.model_name)

        configure_litellm(model_name=self.model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            response = embedding(model=self.embedding_model, input=[text])
            return np.array(response.data[0]['embedding'], dtype=np.float32)
        except Exception as e:
            print(f'Error getting embedding: {e}')
            return np.zeros(1536, dtype=np.float32)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(
        self,
        query: str,
        messages: Dict[str, Any],
        action_history: Any,
        action_space: str,
        use_llm: bool = True,
    ) -> float:
        query = query
        gist = messages['response']

        if use_llm:
            final_relevance = self._ask_llm_relevance(
                query, gist, action_history, action_space
            )
        else:
            final_relevance = self._ask_statistical_relevance(query, gist)

        return float(np.clip(final_relevance, 0.0, 1.0))

    def _ask_llm_relevance(
        self, query: str, gist: str, action_history: Any, action_space: str
    ) -> float:
        relevance_prompt = [
            {
                'role': 'user',
                'content': RELEVANCE_PROMPT_OPTION.format(
                    objective=query,
                    gist=gist,
                    action_history=action_history,
                    action_space=action_space,
                ),
            }
        ]

        try:
            responses = completion(
                messages=relevance_prompt,
                model=self.relevance_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            score_text = responses.choices[0].message.content.strip()

            import re

            number_match = re.search(r'(\d+\.?\d*)', score_text)
            if number_match:
                score = float(number_match.group(1))
            else:
                score = float(score_text)

            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            raise ValueError('Error getting relevance score')

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_confidence(
        self,
        query: str,
        messages: Dict[str, Any],
        action_history: Any,
        action_space: str,
        use_llm: bool = True,
    ) -> float:
        if use_llm:
            final_confidence = self._ask_llm_confidence(
                query, messages, action_history, action_space
            )
        else:
            final_confidence = 0.5

        return float(np.clip(final_confidence, 0.0, 1.0))

    def _ask_llm_confidence(
        self,
        query: str,
        messages: Dict[str, Any],
        action_history: Any,
        action_space: str,
        use_llm: bool = True,
    ) -> float:
        gist = messages['response']
        confidence_prompt = [
            {
                'role': 'user',
                'content': CONFIDENCT_PROMPT_OPTION.format(
                    objective=query,
                    gist=gist,
                    action_history=action_history,
                    action_space=action_space,
                ),
            }
        ]

        try:
            responses = completion(
                messages=confidence_prompt,
                model=self.confidence_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            score_text = responses.choices[0].message.content.strip()

            import re

            number_match = re.search(r'(\d+\.?\d*)', score_text)
            if number_match:
                score = float(number_match.group(1))
            else:
                score = float(score_text)

            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            return 0.5

    def ask(
        self,
        query: str,
        messages: Dict[str, Any],
        action_history: Any,
        action_space: str,
        use_llm: bool = True,
        **kwargs,
    ) -> Dict[str, float]:
        relevance = self.ask_relevance(
            query, messages, action_history, action_space, use_llm=use_llm
        )
        confidence = self.ask_confidence(
            query, messages, action_history, action_space, use_llm=use_llm
        )

        weight = relevance + confidence

        return {
            'relevance': relevance,
            'confidence': confidence,
            'surprise': 0.0,
            'weight': weight,
        }
