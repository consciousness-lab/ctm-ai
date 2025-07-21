from .scorer_base import BaseScorer


@BaseScorer.register_scorer("language_scorer")
class LanguageScorer(BaseScorer):
    def init_scorer(self, *args, **kwargs) -> None:
        super().init_scorer(*args, **kwargs)
