import logging
from functools import wraps
from logging import StreamHandler
from typing import Any, Callable, Dict, List, Literal, Mapping, Union

from termcolor import colored

LogType = Union[List[Dict[str, str]], None]

ColorType = Literal[
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'light_grey',
    'dark_grey',
    'light_red',
    'light_green',
    'light_yellow',
    'light_blue',
    'light_magenta',
    'light_cyan',
    'white',
]

LOG_COLORS: Mapping[str, ColorType] = {
    'BACKGROUND LOG': 'blue',
    'ACTION': 'green',
    'OBSERVATION': 'yellow',
    'DETAIL': 'cyan',
    'ERROR': 'red',
    'PLAN': 'light_magenta',
}


class ColoredFormatter(logging.Formatter):
    def format(self: logging.Formatter, record: logging.LogRecord) -> Any:
        msg_type = record.__dict__.get('msg_type', None)
        if msg_type in LOG_COLORS:
            msg_type_color = colored(msg_type, LOG_COLORS[msg_type])
            msg = colored(record.msg, LOG_COLORS[msg_type])
            time_str = colored(
                self.formatTime(record, self.datefmt), LOG_COLORS[msg_type]
            )
            name_str = colored(record.name, LOG_COLORS[msg_type])
            level_str = colored(record.levelname, LOG_COLORS[msg_type])
            if msg_type == 'ERROR':
                return f'{time_str} - {name_str}:{level_str}: {record.filename}:{record.lineno}\n{msg_type_color}\n{msg}'
            return f'{time_str} - {msg_type_color}\n{msg}'
        elif msg_type == 'STEP':
            msg = '\n\n==============\n' + record.msg + '\n'
            return f'{msg}'
        return logging.Formatter.format(self, record)


console_formatter = ColoredFormatter(
    '\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)


def get_console_handler() -> StreamHandler[Any]:
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    return console_handler


logger = logging.getLogger('CTM-AI')
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())


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
            text = message.get('text', '')
            level = str(message.get('level', 'INFO')).upper()

            if level == 'DEBUG':
                logger.debug(text)
            elif level == 'INFO':
                logger.info(text)
            elif level == 'WARNING':
                logger.warning(text)
            elif level == 'ERROR':
                logger.error(text)
            elif level == 'CRITICAL':
                logger.critical(text)
            else:
                logger.info(text)  # Default to INFO if the level is not recognized

    return wrapper


def logging_ask(
    level: str = 'INFO',
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
            class_name = args[0].__class__.__name__
            result = func(*args, **kwargs)
            log_message = f'Asking {class_name} and return\n{result}'
            getattr(logger, level.lower())(log_message)
            return result

        return wrapper

    return decorator


def logging_chunk(func: Callable[..., Any]) -> Callable[..., None]:
    @wraps(func)
    def wrapper(self: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        func(self, *args, **kwargs)
        logger.info(
            f'{self.processor_name} creates \ngist:\n{self.gist}\nweight:\n{self.weight}'
        )

    return wrapper


def logging_func(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        logger.info(f'========== {func.__name__} starting ==========')
        result = func(self, *args, **kwargs)
        logger.info(f'========== {func.__name__} finished ==========')
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
            f'========== {func.__name__} call #{call_number} starting =========='
        )

        result = func(self, *args, **kwargs)

        logger.info(
            f'========== {func.__name__} call #{call_number} finished =========='
        )
        return result

    return wrapper


def logging_chunk_compete(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, chunk1: Any, chunk2: Any) -> Any:
        logger.info(f'Competing {chunk1.processor_name} vs {chunk2.processor_name}')
        result = func(self, chunk1, chunk2)
        logger.info(f'Winner: {result.processor_name}')
        return result

    return wrapper
