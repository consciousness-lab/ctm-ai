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

    def uptree_competition(self) -> Chunk:
        if not self.chunks:
            raise ValueError('No chunks available for competition')

        if len(self.chunks) == 1:
            return self.chunks[0]

        weights = [self._sanitize_weight(chunk.weight) for chunk in self.chunks]

        total_weight = sum(weights)
        if total_weight == 0:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / total_weight for w in weights]

        winning_chunk = np.random.choice(self.chunks, p=normalized_weights)

        return winning_chunk

    def compete(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        if chunk1.weight > chunk2.weight:
            return chunk1
        elif chunk1.weight < chunk2.weight:
            return chunk2
        else:
            return random.choice([chunk1, chunk2])