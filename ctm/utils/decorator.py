import math
import time
from functools import wraps
from typing import Any, Callable, Optional

INF = float(math.inf)


def info_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    """

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = base_wait_time * (2**attempts)
                    print(f"Attempt {attempts + 1} failed: {e}")
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            print(
                f"Failed to execute '{func.__name__}' after {retries} retries."
            )
            return "FAILED"

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
                    print(f"Attempt {attempts + 1} failed: {e}")
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            print(
                f"Failed to execute '{func.__name__}' after {retries} retries."
            )
            return -INF

        return wrapper

    return decorator
