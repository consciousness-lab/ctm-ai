import random
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import logger, logging_chunk_compete
from .chunk import Chunk


class ChunkManager:
    def __init__(self, chunks: List[Chunk] = []):
        self.chunks: List[Chunk] = chunks
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self._update_tfidf()

    def add_chunk(self, chunk: Chunk):
        self.chunks.append(chunk)
        self._update_tfidf()

    def add_chunks(self, chunks: List[Chunk]):
        self.chunks.extend(chunks)
        self._update_tfidf()

    def remove_chunk(self, index: int):
        self.chunks.pop(index)
        self._update_tfidf()

    def _update_tfidf(self):
        gists = [chunk.gist for chunk in self.chunks if chunk.gist is not None]
        if gists:
            self.tfidf_matrix = self.vectorizer.fit_transform(gists)
        else:
            self.tfidf_matrix = None

    def get_similarity_matrix(self):
        if self.tfidf_matrix is not None:
            return cosine_similarity(self.tfidf_matrix)
        return np.array([])  # Return an empty array if no data is available

    def reset(self):
        """Clears all chunks and resets the TF-IDF matrix."""
        self.chunks.clear()
        self.tfidf_matrix = None

    @logging_chunk_compete
    def compete(self, chunk1: Chunk, chunk2: Chunk):
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
            for chunk1, chunk2 in zip(
                candidate_chunks[:-1], candidate_chunks[1:]
            ):
                winning_chunk = self.compete(chunk1, chunk2)
                winning_chunks.append(winning_chunk)
            candidate_chunks = winning_chunks
        return candidate_chunks[0]
