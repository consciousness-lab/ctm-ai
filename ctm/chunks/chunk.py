from typing import Optional

from ..utils import logging_chunk


class Chunk:
    @logging_chunk
    def __init__(
        self,
        time_step: int,
        processor_name: Optional[str] = None,
        gist: Optional[str] = None,
        relevance: Optional[float] = None,
        confidence: Optional[float] = None,
        surprise: Optional[float] = None,
        weight: Optional[float] = None,
        intensity: Optional[float] = None,
        mood: Optional[float] = None,
    ):
        self.time_step = time_step
        self.processor_name = processor_name
        self.gist = gist
        self.relevance = relevance
        self.confidence = confidence
        self.surprise = surprise
        self.weight = weight
        self.intensity = intensity
        self.mood = mood

    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.weight == other.weight

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.weight < other.weight

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.weight > other.weight

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)
