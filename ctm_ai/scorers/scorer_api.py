import json
from typing import List

from ..messengers import Message
from ..utils import ask_llm_standard, score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer('api_scorer')
class APIScorer(BaseScorer):
    def init_scorer(self, *args, **kwargs) -> None:
        super().init_scorer(*args, **kwargs)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message], use_llm: bool = True) -> float:
        """
        Evaluate relevance for API responses, handling both function calls and regular responses.

        Args:
            messages: List of messages containing query and gist
            use_llm: If True, use LLM for relevance scoring; if False, use statistical methods
        """
        if not messages or not messages[-1].query or not messages[-1].gist:
            return 0.0

        query = messages[-1].query
        gist = messages[-1].gist

        # Check if this is a function call response
        is_function_call = self._is_function_call_response(gist)
        breakpoint()

        if use_llm:
            if is_function_call:
                final_relevance = self._ask_llm_relevance_function_call(query, gist)
            else:
                final_relevance = self._ask_llm_relevance_regular(query, gist)
        else:
            # Use statistical method from base class
            final_relevance = self._ask_statistical_relevance(query, gist)

        return float(max(0.0, min(1.0, final_relevance)))

    def _is_function_call_response(self, gist: str) -> bool:
        """
        Determine if the response is a function call by checking if it's JSON-like
        and contains function arguments.
        """
        # First try to parse as JSON
        try:
            parsed = json.loads(gist)
            if isinstance(parsed, dict):
                return True
        except (json.JSONDecodeError, TypeError):
            pass

        # Then try to evaluate as Python dict literal (for str() output)
        try:
            # Only evaluate if it looks like a dict and contains safe characters
            stripped = gist.strip()
            if (
                stripped.startswith('{')
                and stripped.endswith('}')
                and all(c not in stripped for c in ['__', 'import', 'exec', 'eval'])
            ):
                parsed = eval(stripped)
                if isinstance(parsed, dict):
                    return True
        except (SyntaxError, NameError, TypeError, ValueError):
            pass

        # Check if it's a string that looks like function arguments
        if '{' in gist and '}' in gist and ':' in gist:
            return True

        return False

    def _ask_llm_relevance_function_call(self, query: str, gist: str) -> float:
        """
        Evaluate relevance for function call responses.
        """
        # Try to parse and format function call arguments
        parsed_args = None

        if not isinstance(gist, str):
            parsed_args = gist
        else:
            # First try JSON parsing
            try:
                parsed_args = json.loads(gist)
            except (json.JSONDecodeError, TypeError):
                # Then try eval for Python dict literals (safe evaluation)
                try:
                    stripped = gist.strip()
                    if (
                        stripped.startswith('{')
                        and stripped.endswith('}')
                        and all(
                            c not in stripped for c in ['__', 'import', 'exec', 'eval']
                        )
                    ):
                        parsed_args = eval(stripped)
                except (SyntaxError, NameError, TypeError, ValueError):
                    pass

        # Format the arguments for display
        try:
            if parsed_args is not None and isinstance(parsed_args, dict):
                formatted_args = json.dumps(parsed_args, indent=2, ensure_ascii=False)
            else:
                formatted_args = str(gist)
        except (TypeError, ValueError):
            formatted_args = str(gist)

        relevance_prompt = [
            Message(
                role='user',
                content=f"""Please evaluate how relevant the function call is to the user's question on a scale from 0.0 to 1.0.

Question: {query}
Function Call Arguments: {formatted_args}

Consider:
- 1.0 = Function call perfectly matches the user's intent, all required parameters are provided correctly
- 0.8 = Function call mostly matches the intent, most important parameters are correct
- 0.6 = Function call partially matches, some relevant parameters but may miss key details
- 0.4 = Function call somewhat related but parameters don't fully address the question
- 0.2 = Function call barely relevant, weak connection to the user's intent
- 0.0 = Function call completely unrelated or has completely wrong parameters

IMPORTANT CONSIDERATIONS FOR FUNCTION CALLS:
- Focus on whether the function being called is appropriate for the user's question
- Check if the parameters/arguments provided are relevant and correct for the user's intent
- Consider if the function call would logically lead to answering the user's question
- Even if parameters are partially filled, score higher if the function choice is correct

Respond with only a number between 0.0 and 1.0 (e.g., 0.85).""",
            )
        ]

        try:
            responses = ask_llm_standard(
                messages=relevance_prompt,
                model=self.relevance_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            # Extract numerical score
            score_text = responses[0].strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except (ValueError, TypeError, IndexError):
            return 0.5  # Default neutral relevance

    def _ask_llm_relevance_regular(self, query: str, gist: str) -> float:
        """
        Evaluate relevance for regular API responses (non-function calls).
        """
        relevance_prompt = [
            Message(
                role='user',
                content=f"""Please evaluate how relevant the API response is to the user's question on a scale from 0.0 to 1.0.

Question: {query}
API Response: {gist}

Consider:
- 1.0 = Perfectly relevant, directly answers the question with specific information from API
- 0.8 = Highly relevant, mostly answers the question with useful API data
- 0.6 = Moderately relevant, partially answers the question with some API information
- 0.4 = Somewhat relevant, tangentially related API response but not very helpful
- 0.2 = Barely relevant, weak connection or very general API response
- 0.0 = Not relevant, API error, says "cannot determine", or completely unrelated

IMPORTANT: 
- If the response indicates an API error, timeout, or failure, score it as 0.2 or lower
- If the response says "I cannot determine", "I don't know", or refuses to provide information, score it as 0.0
- Focus on whether the API actually provided useful data to answer the user's question

Respond with only a number between 0.0 and 1.0 (e.g., 0.85).""",
            )
        ]

        try:
            responses = ask_llm_standard(
                messages=relevance_prompt,
                model=self.relevance_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            # Extract numerical score
            score_text = responses[0].strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except (ValueError, TypeError, IndexError):
            return 0.5  # Default neutral relevance
