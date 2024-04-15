import math
import time
from functools import wraps


def exponential_backoff(retries=5, base_wait_time=1):  # type: ignore[no-untyped-def] # FIX ME
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    """

    def decorator(func):  # type: ignore[no-untyped-def] # FIX ME
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
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
            return None

        return wrapper

    return decorator
