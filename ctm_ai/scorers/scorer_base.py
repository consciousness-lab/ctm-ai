from typing import Any, Callable, Dict, List, Optional, Tuple, Type


class BaseScorer(object):
    _scorer_registry: Dict[str, Type["BaseScorer"]] = {}

    @classmethod
    def register_scorer(
        cls, scorer_name: str
    ) -> Callable[[Type["BaseScorer"]], Type["BaseScorer"]]:
        def decorator(
            subclass: Type["BaseScorer"],
        ) -> Type["BaseScorer"]:
            cls._scorer_registry[scorer_name] = subclass
            return subclass

        return decorator

    def __new__(
        cls, scorer_name: str, *args: Any, **kwargs: Any
    ) -> "BaseScorer":
        if scorer_name not in cls._scorer_registry:
            raise ValueError(f"No scorer registered with name '{scorer_name}'")
        return super(BaseScorer, cls).__new__(
            cls._scorer_registry[scorer_name]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_scorer()

    def init_scorer(self) -> None:
        raise NotImplementedError(
            "The 'init_scorer' method must be implemented in derived classes."
        )

    def ask_relevance(self, query: str, gist: str) -> float:
        raise NotImplementedError(
            "The 'ask_relevance' method must be implemented in derived classes."
        )

    def ask_confidence(self, query: str, gist: str) -> float:
        raise NotImplementedError(
            "The 'ask_confidence' method must be implemented in derived classes."
        )

    def ask_surprise(
        self, query: str, gist: str, history_gists: Optional[str] = None
    ) -> float:
        raise NotImplementedError(
            "The 'ask_surprise' method must be implemented in derived classes."
        )

    def ask(
        self,
        query: str,
        gist: str,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[float, float, float]:
        relevance = self.ask_relevance(query, gist, *args, **kwargs)
        confidence = self.ask_confidence(query, gist, *args, **kwargs)
        surprise = self.ask_surprise(query, gist, *args, **kwargs)
        weight = relevance * confidence * surprise
        if verbose:
            print(
                f"Relevance: {relevance}, Confidence: {confidence}, Surprise: {surprise}"
            )
        score = {
            "relevance": relevance,
            "confidence": confidence,
            "surprise": surprise,
            "weight": weight,
        }
        return score
