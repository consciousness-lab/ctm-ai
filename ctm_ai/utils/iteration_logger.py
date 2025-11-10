import json
import threading
from datetime import datetime
from functools import wraps
from threading import Lock
from typing import Any, Dict, List, Optional


class IterationLogger:
    """Thread-safe logger for CTM iteration information."""

    def __init__(self, output_file: str = 'ctm_iterations.jsonl'):
        self.output_file = output_file
        self.current_test_file = None
        self.current_iteration = 0
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._lock = Lock()

    def set_test_file(self, test_file: str):
        """Set the current test file being processed."""
        self.current_test_file = test_file
        self.current_iteration = 0

    def log_iteration(
        self,
        iteration: int,
        winning_chunk: Any,
        all_chunks: List[Any],
        query: str,
        confidence_score: float,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Log information for one iteration."""

        # Extract information from chunks
        chunk_info = []
        for i, chunk in enumerate(all_chunks, 1):
            chunk_data = {
                'rank': i,
                'processor_name': getattr(chunk, 'processor_name', 'unknown'),
                'gist': getattr(chunk, 'gist', ''),
                'additional_question': getattr(chunk, 'additional_question', ''),
                'relevance': getattr(chunk, 'relevance', 0.0),
                'confidence': getattr(chunk, 'confidence', 0.0),
                'surprise': getattr(chunk, 'surprise', 0.0),
                'weight': getattr(chunk, 'weight', 0.0),
            }
            chunk_info.append(chunk_data)

        # Log entry
        log_entry = {
            'session_id': self.session_id,
            'test_file': self.current_test_file,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'winning_chunk': {
                'processor_name': getattr(winning_chunk, 'processor_name', 'unknown'),
                'gist': getattr(winning_chunk, 'gist', ''),
                'additional_question': getattr(
                    winning_chunk, 'additional_question', ''
                ),
                'relevance': getattr(winning_chunk, 'relevance', 0.0),
                'confidence': getattr(winning_chunk, 'confidence', 0.0),
                'surprise': getattr(winning_chunk, 'surprise', 0.0),
                'weight': getattr(winning_chunk, 'weight', 0.0),
            },
            'all_chunks': chunk_info,
            'confidence_score': confidence_score,
            'total_chunks': len(all_chunks),
        }

        # Add additional info if provided
        if additional_info:
            log_entry.update(additional_info)

        # Write to file with thread safety
        with self._lock:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


# Thread-local storage for loggers
_thread_local = threading.local()


def set_iteration_log_file(test_file: str, output_file: str = 'ctm_iterations.jsonl'):
    """Set the current test file and output file for logging (thread-safe)."""
    logger = IterationLogger(output_file)
    logger.set_test_file(test_file)
    _thread_local.logger = logger


def _get_logger():
    """Get the thread-local logger instance."""
    if not hasattr(_thread_local, 'logger'):
        # Fallback to default logger if not set
        _thread_local.logger = IterationLogger()
    return _thread_local.logger


def log_ctm_iteration(func):
    """Decorator to log CTM iteration information."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original function
        result = func(self, *args, **kwargs)

        # Extract information for logging
        if hasattr(self, '_current_iteration_info'):
            info = self._current_iteration_info
            logger = _get_logger()
            logger.log_iteration(
                iteration=info.get('iteration', 0),
                winning_chunk=info.get('winning_chunk'),
                all_chunks=info.get('all_chunks', []),
                query=info.get('query', ''),
                confidence_score=info.get('confidence_score', 0.0),
                additional_info=info.get('additional_info', {}),
            )
            # Clear the info after logging
            delattr(self, '_current_iteration_info')

        return result

    return wrapper


def log_go_up_iteration(func):
    """Decorator specifically for go_up method to capture iteration data."""

    @wraps(func)
    def wrapper(self, query: str, **input_kwargs):
        # Call the original function
        winning_chunk, chunks = func(self, query, **input_kwargs)

        # Store iteration info for later logging
        if not hasattr(self, '_iteration_counter'):
            self._iteration_counter = 0
        self._iteration_counter += 1

        self._current_winning_chunk = winning_chunk
        self._current_chunks = chunks
        self._current_query = query
        self._current_iteration = self._iteration_counter

        return winning_chunk, chunks

    return wrapper


def log_forward_iteration(func):
    """Decorator for forward method to log complete iteration information."""

    @wraps(func)
    def wrapper(self, query: str, *args, **kwargs):
        # Reset iteration counter
        self._iteration_counter = 0

        # Call the original function
        result = func(self, query, *args, **kwargs)

        return result

    return wrapper


def log_supervisor_result(func):
    """Decorator to capture supervisor results and trigger logging."""

    @wraps(func)
    def wrapper(self, query: str, chunk, *args, **kwargs):
        # Call the original function
        answer, confidence_score = func(self, query, chunk, *args, **kwargs)

        # If we have stored iteration data, log it now
        if (
            hasattr(self, '_current_winning_chunk')
            and hasattr(self, '_current_chunks')
            and hasattr(self, '_current_iteration')
        ):
            logger = _get_logger()
            logger.log_iteration(
                iteration=self._current_iteration,
                winning_chunk=self._current_winning_chunk,
                all_chunks=self._current_chunks,
                query=query,
                confidence_score=confidence_score,
                additional_info={
                    'supervisor_answer': answer,
                },
            )

        return answer, confidence_score

    return wrapper
