from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..chunks.chunk import Chunk


def calc_chunk_sim(chunks: List[Chunk]) -> List[List[float]]:
    gists = [chunk.gist for chunk in chunks]
    tfidf = TfidfVectorizer().fit_transform(gists)
    cos_sim = cosine_similarity(tfidf, tfidf)
    return cos_sim
