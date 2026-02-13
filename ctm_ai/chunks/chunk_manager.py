import math
import random
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..configs import ConsciousTuringMachineConfig
from .chunk import Chunk


class ChunkManager:
    def __init__(
        self,
        chunks: List[Chunk] = [],
        config: Optional[ConsciousTuringMachineConfig] = None,
    ) -> None:
        self.config = config
        self.chunks: List[Chunk] = chunks
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix: Optional[NDArray[np.float32]] = None
        self._update_tfidf()

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        self._update_tfidf()

    def add_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._update_tfidf()

    def remove_chunk(self, index: int) -> None:
        self.chunks.pop(index)
        self._update_tfidf()

    def _update_tfidf(self) -> None:
        gists = [chunk.gist for chunk in self.chunks if chunk.gist]
        if gists:
            self.tfidf_matrix = self.vectorizer.fit_transform(gists).toarray()
        else:
            self.tfidf_matrix = None

    def _sanitize_weight(self, weight: float) -> float:
        if math.isnan(weight):
            return 0.0
        return weight

    def _get_similarity_matrix(self) -> NDArray[np.float32]:
        if self.tfidf_matrix is not None:
            return cosine_similarity(self.tfidf_matrix)
        return np.array([])

    def reset(self) -> None:
        """Clears all chunks and resets the TF-IDF matrix."""
        self.chunks.clear()
        self.tfidf_matrix = None

    def uptree_competition(self, temperature: float = 0.1) -> Chunk:
        """
        Weighted random selection using softmax with temperature.

        Args:
            temperature: Temperature parameter for softmax (default=0.1).
                   - temperature→0: deterministic, always select max weight (argmax)
                   - temperature=0.1: strongly favor high weights (recommended)
                   - temperature=1.0: standard softmax
                   - temperature→∞: uniform distribution
        """
        if not self.chunks:
            raise ValueError('No chunks available for competition')

        if len(self.chunks) == 1:
            return self.chunks[0]

        weights = np.array(
            [self._sanitize_weight(chunk.weight) for chunk in self.chunks]
        )

        # 防止除以零
        if temperature <= 0:
            temperature = 1e-10

        # Softmax with temperature: exp(w/T) / sum(exp(w/T))
        # 为了数值稳定性，减去最大值
        weights_scaled = weights / temperature
        weights_shifted = weights_scaled - np.max(weights_scaled)
        exp_weights = np.exp(weights_shifted)

        total_exp = np.sum(exp_weights)
        if total_exp == 0 or not np.isfinite(total_exp):
            normalized_weights = np.ones(len(weights)) / len(weights)
        else:
            normalized_weights = exp_weights / total_exp

        winning_chunk = np.random.choice(self.chunks, p=normalized_weights)

        return winning_chunk

    def compete(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        if chunk1.weight > chunk2.weight:
            return chunk1
        elif chunk1.weight < chunk2.weight:
            return chunk2
        else:
            return random.choice([chunk1, chunk2])
