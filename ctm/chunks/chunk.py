from typing import Optional


class Chunk(object):
    def __init__(
        self,
        processor_name: str = None,
        time_step: int = -1,
        gist: Optional[str] = None,
        relevance: Optional[float] = None,
        confidence: Optional[float] = None,
        surprise: Optional[float] = None,
        weight: Optional[float] = None,
        intensity: Optional[float] = None,
        mood: Optional[float] = None,
    ):
        self.processor_name: Optional[str] = processor_name
        self.time_step: int = time_step
        self.gist: Optional[str] = gist
        self.relevance: Optional[float] = relevance
        self.confidence: Optional[float] = confidence
        self.surprise: Optional[float] = surprise
        self.weight: Optional[float] = weight
        self.intensity: Optional[float] = intensity
        self.mood: Optional[float] = mood

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
