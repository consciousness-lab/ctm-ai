import json
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chunks import Chunk


class InstanceLogger:
    """Logger for recording CTM forward pass information per instance."""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.short_logs: List[Dict[str, Any]] = []
        self.long_logs: List[Dict[str, Any]] = []
        self.current_iteration: int = 0
        self.instance_id: str = ''
        self._enabled: bool = True

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def reset(self, instance_id: Optional[str] = None) -> None:
        self.short_logs = []
        self.long_logs = []
        self.current_iteration = 0
        if instance_id:
            self.instance_id = instance_id
        else:
            self.instance_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def set_iteration(self, iteration: int) -> None:
        self.current_iteration = iteration

    def log_ask_processors(self, chunks: List[Chunk]) -> None:
        if not self._enabled:
            return

        # Short log: processor names
        self.short_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'ask_processors',
                'processors': [chunk.processor_name for chunk in chunks],
            }
        )

        # Long log: all chunk details
        self.long_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'ask_processors',
                'chunks': [
                    {
                        'processor_name': chunk.processor_name,
                        'gist': chunk.gist,
                        'weight': chunk.weight,
                        'relevance': chunk.relevance,
                        'additional_question': chunk.additional_question,
                    }
                    for chunk in chunks
                ],
            }
        )

    def log_uptree_competition(
        self, winning_chunk: Chunk, all_chunks: List[Chunk]
    ) -> None:
        if not self._enabled:
            return

        # Short log
        self.short_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'uptree_competition',
                'winning_processor': winning_chunk.processor_name,
                'winning_weight': winning_chunk.weight,
            }
        )

        # Long log
        self.long_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'uptree_competition',
                'winning_chunk': {
                    'processor_name': winning_chunk.processor_name,
                    'gist': winning_chunk.gist,
                    'weight': winning_chunk.weight,
                    'relevance': winning_chunk.relevance,
                    'additional_question': winning_chunk.additional_question,
                },
                'all_weights': {
                    chunk.processor_name: chunk.weight for chunk in all_chunks
                },
            }
        )

    def log_supervisor(self, answer: str, confidence_score: float) -> None:
        if not self._enabled:
            return

        # Short log
        self.short_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'supervisor',
                'confidence_score': confidence_score,
            }
        )

        # Long log
        self.long_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'supervisor',
                'answer': answer,
                'confidence_score': confidence_score,
            }
        )

    def log_downtree_broadcast(self, winning_chunk: Chunk) -> None:
        if not self._enabled:
            return

        self.short_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'downtree_broadcast',
                'broadcast_from': winning_chunk.processor_name,
            }
        )

        self.long_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'downtree_broadcast',
                'broadcast_chunk': {
                    'processor_name': winning_chunk.processor_name,
                    'gist': winning_chunk.gist,
                },
            }
        )

    def log_link_form(
        self,
        winning_chunk: Chunk,
        added_links: List[tuple],
        removed_links: List[tuple],
    ) -> None:
        if not self._enabled:
            return

        # Short log
        self.short_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'link_form',
                'source_processor': winning_chunk.processor_name,
                'added_links': [link[1] for link in added_links],
                'removed_links': [link[1] for link in removed_links],
            }
        )

        # Long log
        self.long_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'link_form',
                'source_processor': winning_chunk.processor_name,
                'additional_question': winning_chunk.additional_question,
                'added_links': added_links,
                'removed_links': removed_links,
            }
        )

    def log_fuse(
        self,
        dirty_processors: List[str],
        fuse_info: List[Dict[str, Any]],
    ) -> None:
        if not self._enabled:
            return

        # Short log
        self.short_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'fuse',
                'dirty_processors': dirty_processors,
            }
        )

        # Long log
        self.long_logs.append(
            {
                'iteration': self.current_iteration,
                'event': 'fuse',
                'dirty_processors': dirty_processors,
                'fuse_details': fuse_info,
            }
        )

    def save(self) -> None:
        if not self._enabled:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save short log
        short_log_path = self.log_dir / f'{self.instance_id}_short.jsonl'
        with open(short_log_path, 'w') as f:
            for entry in self.short_logs:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # Save long log
        long_log_path = self.log_dir / f'{self.instance_id}_long.jsonl'
        with open(long_log_path, 'w') as f:
            for entry in self.long_logs:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# Global instance logger
_instance_logger: Optional[InstanceLogger] = None


def get_instance_logger() -> InstanceLogger:
    global _instance_logger
    if _instance_logger is None:
        _instance_logger = InstanceLogger()
    return _instance_logger


def set_instance_logger(logger: InstanceLogger) -> None:
    global _instance_logger
    _instance_logger = logger


def log_forward(func):
    """Decorator to wrap forward method with instance logging."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = get_instance_logger()
        instance_id = kwargs.pop('instance_id', None)
        logger.reset(instance_id)

        result = func(self, *args, **kwargs)

        logger.save()
        return result

    return wrapper
