import random
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..configs import BaseConsciousnessTuringMachineConfig
from ..utils import logging_chunk_compete
from .chunk import Chunk


class ChunkManager:
    def __init__(
        self,
        chunks: List[Chunk] = [],
        config: Optional[BaseConsciousnessTuringMachineConfig] = None,
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

    def _get_similarity_matrix(self) -> NDArray[np.float32]:
        if self.tfidf_matrix is not None:
            return cosine_similarity(self.tfidf_matrix)
        return np.array([])  # Return an empty array if no data is available

    def get_interaction_type_matrix(self) -> NDArray[np.float32]:
        sim = self._get_similarity_matrix()
        interaction_type_matrix = np.zeros_like(sim)

        if self.config is None:
            raise ValueError(
                'Config must be provided for interaction type calculation.'
            )

        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                chunk_i, chunk_j = self.chunks[i], self.chunks[j]

                # Check for redundant interaction
                if (
                    sim[i][j] > self.config.redundant_text_sim_threshold
                    and chunk_i.weight > self.config.redundant_weight_threshold
                    and chunk_j.weight > self.config.redundant_weight_threshold
                ):
                    interaction_type_matrix[i][j] = 1
                # Check for synergy interaction
                elif (
                    sim[i][j] < self.config.synergy_text_sim_threshold
                    and chunk_i.weight > self.config.synergy_weight_threshold
                    and chunk_j.weight > self.config.synergy_weight_threshold
                ):
                    interaction_type_matrix[i][j] = -1
        return interaction_type_matrix

    def reset(self) -> None:
        """Clears all chunks and resets the TF-IDF matrix."""
        self.chunks.clear()
        self.tfidf_matrix = None

    @logging_chunk_compete
    def compete(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        if chunk1 > chunk2:
            winner = chunk1
        elif chunk1 < chunk2:
            winner = chunk2
        else:
            winner = random.choice([chunk1, chunk2])
        return winner

    def uptree_competition(self) -> Chunk:
        candidate_chunks: List[Chunk] = self.chunks
        while len(candidate_chunks) > 1:
            winning_chunks: List[Chunk] = []
            for chunk1, chunk2 in zip(candidate_chunks[:-1], candidate_chunks[1:]):
                winning_chunk = self.compete(chunk1, chunk2)
                winning_chunks.append(winning_chunk)
            candidate_chunks = winning_chunks
        return candidate_chunks[0]
