from typing import Any, Callable, Dict, List, Type

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import word_frequency

from ..messengers import Message
from ..utils import score_exponential_backoff


class BaseScorer(object):
    _scorer_registry: Dict[str, Type['BaseScorer']] = {}

    @classmethod
    def register_scorer(
        cls, name: str
    ) -> Callable[[Type['BaseScorer']], Type['BaseScorer']]:
        def decorator(
            subclass: Type['BaseScorer'],
        ) -> Type['BaseScorer']:
            cls._scorer_registry[name] = subclass
            return subclass

        return decorator

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> 'BaseScorer':
        if name not in cls._scorer_registry:
            raise ValueError(f"No scorer registered with name '{name}'")
        instance = super(BaseScorer, cls).__new__(cls._scorer_registry[name])
        instance.name = name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_scorer()

    def init_scorer(self) -> None:
        raise NotImplementedError(
            "The 'init_scorer' method must be implemented in derived classes."
        )

    def ask_relevance(self, messages: List[Message]) -> float:
        raise NotImplementedError(
            "The 'ask_relevance' method must be implemented in derived classes."
        )

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_confidence(self, messages: List[Message]) -> float:
        gist = messages[-1].gist
        gists = messages[-1].gists
        if gist is None or gist == '':
            return 0.0
        if gists == []:
            return 1.0

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(gists)
        cos_sim_matrix = cosine_similarity(tfidf_matrix)

        upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
        upper_triangle_values = cos_sim_matrix[upper_triangle_indices]

        avg_cos_sim = np.mean(upper_triangle_values)

        confidence = float(avg_cos_sim)
        return confidence

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_surprise(
        self,
        messages: List[Message],
        lang: str = 'en',
    ) -> float:
        if messages[-1].gist is None or messages[-1].gist == '':
            return 0.0
        gist_words = messages[-1].gist.split()
        word_freqs = [
            float(word_frequency(gist_word, lang)) for gist_word in gist_words
        ]
        surprise = sum(word_freqs) / len(word_freqs) if word_freqs else 0
        return surprise

    def ask(
        self,
        messages: List[Message],
    ) -> Message:
        
        relevance = self.ask_relevance(messages)
        confidence = self.ask_confidence(messages)
        surprise = self.ask_surprise(messages)
        weight = relevance * confidence * surprise
        message = Message(
            relevance=relevance,
            confidence=confidence,
            surprise=surprise,
            weight=weight,
        )
        return message
