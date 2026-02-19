import json
import logging
import os
from functools import wraps
from logging import StreamHandler
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from termcolor import colored

LogType = Union[List[Dict[str, str]], None]

# Global dictionary to store iteration log files per query_id
_iteration_log_files: Dict[str, str] = {}

ColorType = Literal[
    "grey",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
]

LOG_COLORS: Mapping[str, ColorType] = {
    "BACKGROUND LOG": "blue",
    "ACTION": "green",
    "OBSERVATION": "yellow",
    "DETAIL": "cyan",
    "ERROR": "red",
    "PLAN": "light_magenta",
}


class ColoredFormatter(logging.Formatter):
    def format(self: logging.Formatter, record: logging.LogRecord) -> Any:
        msg_type = record.__dict__.get("msg_type", None)
        if msg_type in LOG_COLORS:
            msg_type_color = colored(msg_type, LOG_COLORS[msg_type])
            msg = colored(record.msg, LOG_COLORS[msg_type])
            time_str = colored(
                self.formatTime(record, self.datefmt), LOG_COLORS[msg_type]
            )
            name_str = colored(record.name, LOG_COLORS[msg_type])
            level_str = colored(record.levelname, LOG_COLORS[msg_type])
            if msg_type == "ERROR":
                return f"{time_str} - {name_str}:{level_str}: {record.filename}:{record.lineno}\n{msg_type_color}\n{msg}"
            return f"{time_str} - {msg_type_color}\n{msg}"
        elif msg_type == "STEP":
            msg = "\n\n==============\n" + record.msg + "\n"
            return f"{msg}"
        return logging.Formatter.format(self, record)


console_formatter = ColoredFormatter(
    "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
    datefmt="%H:%M:%S",
)


def get_console_handler() -> StreamHandler:  # type: ignore
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    return console_handler


logger = logging.getLogger("CTM-AI")
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())

log_file = "ctm_log_output.log"
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s",
    datefmt="%H:%M:%S",
)
file_handler.setFormatter(file_formatter)

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)


def logging_decorator(
    func: Callable[..., LogType],
) -> Callable[..., None]:
    def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> None:
        messages = func(*args, **kwargs)
        if not messages:
            return
        for message in messages:
            import pdb

            pdb.set_trace()
            text = message.get("text", "")
            level = str(message.get("level", "INFO")).upper()

            if level == "DEBUG":
                logger.debug(text)
            elif level == "INFO":
                logger.info(text)
            elif level == "WARNING":
                logger.warning(text)
            elif level == "ERROR":
                logger.error(text)
            elif level == "CRITICAL":
                logger.critical(text)
            else:
                logger.info(text)  # Default to INFO if the level is not recognized

    return wrapper


def logging_ask(
    level: str = "INFO",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
            class_name = args[0].__class__.__name__
            result = func(*args, **kwargs)
            log_message = f"Asking {class_name} and return\n{result}"
            getattr(logger, level.lower())(log_message)
            return result

        return wrapper

    return decorator


def logging_chunk(func: Callable[..., Any]) -> Callable[..., None]:
    @wraps(func)
    def wrapper(self: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        func(self, *args, **kwargs)
        questions_str = (
            ", ".join(self.additional_questions)
            if self.additional_questions
            else "None"
        )
        logger.info(
            f"{self.processor_name} creates \ngist:\n{self.gist}\nadditional_questions:\n{questions_str}\nweight:\n{self.weight}\nrelevance:\n{self.relevance}\nconfidence:\n{self.confidence}\nsurprise:\n{self.surprise}"
        )

    return wrapper


def logging_func(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        logger.info(f"========== {func.__name__} starting ==========")
        result = func(self, *args, **kwargs)
        logger.info(f"========== {func.__name__} finished ==========")
        return result

    return wrapper


def logging_func_with_count(func: Callable[..., Any]) -> Callable[..., Any]:
    call_count = 0

    @wraps(func)
    def wrapper(self: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        nonlocal call_count
        call_count += 1
        call_number = call_count
        logger.info(
            f"========== {func.__name__} call #{call_number} starting =========="
        )

        result = func(self, *args, **kwargs)

        logger.info(
            f"========== {func.__name__} call #{call_number} finished =========="
        )
        return result

    return wrapper


def logging_chunk_compete(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, chunk1: Any, chunk2: Any) -> Any:
        logger.info(f"Competing {chunk1.processor_name} vs {chunk2.processor_name}")
        result = func(self, chunk1, chunk2)
        logger.info(f"Winner: {result.processor_name}")
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Iteration logging for ToolBench
# ---------------------------------------------------------------------------


def set_iteration_log_file(query_id: str, log_file: str) -> None:
    """
    Set the log file path for a specific query_id.
    Used for ToolBench to track iterations per query.
    """
    global _iteration_log_files
    _iteration_log_files[query_id] = log_file

    # Create directory if not exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def get_iteration_log_file(query_id: str) -> Optional[str]:
    """Get the log file path for a specific query_id."""
    return _iteration_log_files.get(query_id)


def log_iteration(query_id: str, iteration_data: Dict[str, Any]) -> None:
    """Log an iteration's data to the corresponding log file."""
    log_file = get_iteration_log_file(query_id)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(iteration_data, ensure_ascii=False) + "\n")


def log_go_up_iteration(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log go_up phase iterations."""

    @wraps(func)
    def wrapper(self: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        logger.info("========== go_up iteration starting ==========")
        result = func(self, *args, **kwargs)
        logger.info("========== go_up iteration finished ==========")
        return result

    return wrapper


def log_forward_iteration(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log forward iterations."""

    @wraps(func)
    def wrapper(self: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        logger.info("========== forward starting ==========")
        result = func(self, *args, **kwargs)
        logger.info("========== forward finished ==========")
        return result

    return wrapper
