from ..utils import logging_chunk
from typing import Any, Dict


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

    def serialize(self) -> Dict[str, Any]:
        return {
            'time_step': self.time_step,
            'processor_name': self.processor_name,
            'gist': self.gist,
            'relevance': self.relevance,
            'confidence': self.confidence,
            'surprise': self.surprise,
            'weight': self.weight,
            'intensity': self.intensity,
            'mood': self.mood,
        }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'Chunk':
        return Chunk(
            time_step=data['time_step'],
            processor_name=data['processor_name'],
            gist=data['gist'],
            relevance=data['relevance'],
            confidence=data['confidence'],
            surprise=data['surprise'],
            weight=data['weight'],
            intensity=data['intensity'],
            mood=data['mood'],
        )