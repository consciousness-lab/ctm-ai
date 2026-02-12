from typing import Any, Dict, List

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
        additional_questions: List[str] = None,
        executor_content: str = '',
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
        self.additional_questions: List[str] = additional_questions or []
        self.executor_content: str = executor_content

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
            'additional_questions': self.additional_questions,
        }

    def format_readable(self) -> str:
        questions_str = (
            '\n  '.join(self.additional_questions)
            if self.additional_questions
            else 'None'
        )
        return (
            # f'Time Step: {self.time_step}\n'
            f'Processor Name: {self.processor_name}\n'
            f'Gist: {self.gist}\n'
            f'Relevance: {self.relevance:.2f}\n'
            f'Confidence: {self.confidence:.2f}\n'
            f'Surprise: {self.surprise:.2f}\n'
            f'Weight: {self.weight:.2f}\n'
            # f'Intensity: {self.intensity:.2f}\n'
            # f'Mood: {self.mood:.2f}\n'
            f'Additional Questions:\n  {questions_str}'
        )

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'Chunk':
        # Handle backward compatibility with old 'additional_question' field
        additional_questions = data.get('additional_questions', [])
        if not additional_questions and 'additional_question' in data:
            old_q = data.get('additional_question', '')
            additional_questions = [old_q] if old_q else []
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
            additional_questions=additional_questions,
        )
