import math
import time
from functools import wraps
from typing import Any, Callable, List, Union, Dict

from .logger import logger

INF = float(math.inf)


def multi_info_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[
    [Callable[..., List[Union[str, None]]]], Callable[..., List[Union[str, None]]]
]:
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    """

    def decorator(
        func: Callable[..., List[Union[str, None]]],
    ) -> Callable[..., List[Union[str, None]]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> List[Union[str, None]]:
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = base_wait_time * (2**attempts)
                    logger.error(f"Attempt {attempts + 1} failed: {e}")
                    logger.error(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            logger.error(
                f"Failed to execute '{func.__name__}' after {retries} retries.",
            )
            return []

        return wrapper

    return decorator


def info_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[[Callable[..., Union[str, None]]], Callable[..., Union[str, None]]]:
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    """

    def decorator(
        func: Callable[..., Union[str, None]],
    ) -> Callable[..., Union[str, None]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[str, None]:
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = base_wait_time * (2**attempts)
                    logger.error(f"Attempt {attempts + 1} failed: {e}")
                    logger.error(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            logger.error(
                f"Failed to execute '{func.__name__}' after {retries} retries.",
            )
            return None

        return wrapper

    return decorator


def message_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    """

    def decorator(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = base_wait_time * (2**attempts)
                    logger.error(f"Attempt {attempts + 1} failed: {e}")
                    logger.error(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            logger.error(
                f"Failed to execute '{func.__name__}' after {retries} retries.",
            )
            return Dict[str, Any]

        return wrapper

    return decorator


def score_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    """

    def decorator(func: Callable[..., float]) -> Callable[..., float]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> float:
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = base_wait_time * (2**attempts)
                    logger.error(f"Attempt {attempts + 1} failed: {e}")
                    logger.error(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            logger.error(
                f"Failed to execute '{func.__name__}' after {retries} retries."
            )
            return -INF

        return wrapper

    return decorator


class MissingAPIKeyError(Exception):
    def __init__(self, processor_name: str, missing_keys: List[str]):
        self.processor_name = processor_name
        self.missing_keys = missing_keys
        message = (
            f"Processor '{processor_name}' dose not have an API key: "
            f'{", ".join(missing_keys)}'
        )
        super().__init__(message)
