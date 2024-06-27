from ..utils import logging_chunk


class Chunk:
    @logging_chunk
    def __init__(
        self,
        time_step: int,
        processor_name: str,
        gist: str = '',
        relevance: float = -1.0,
        confidence: float = -1.0,
        surprise: float = -1.0,
        weight: float = -1.0,
        intensity: float = -1.0,
        mood: float = -1.0,
    ) -> None:
        self.time_step: int = time_step
        self.processor_name: str = processor_name
        self.gist: str = gist
        self.relevance: float = relevance
        self.confidence: float = confidence
        self.surprise: float = surprise
        self.weight: float = weight
        self.intensity: float = intensity
        self.mood: float = mood

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.weight == other.weight

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.weight < other.weight

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.weight > other.weight

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.__gt__(other) or self.__eq__(other)
