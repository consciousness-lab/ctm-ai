import logging
from functools import wraps
from logging import StreamHandler
from typing import Any, Callable, Dict, List, Mapping, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from termcolor import colored

LogType = Union[List[Dict[str, str]], None]

ColorType = Literal[
    'grey',
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
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


def get_console_handler() -> StreamHandler:  # type: ignore
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    return console_handler


logger = logging.getLogger('CTM-AI')
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())

log_file = 'ctm_log_output.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
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
            f'{self.processor_name} creates \ngist:\n{self.gist}\nadditional_question:\n{self.additional_question}\nweight:\n{self.weight}\nrelevance:\n{self.relevance}\nconfidence:\n{self.confidence}\nsurprise:\n{self.surprise}'
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


def logging_link_form(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(
        self: Any, chunks: List[Any], winning_chunk: Any, **input_kwargs
    ) -> Any:
        additional_question = winning_chunk.additional_question
        logger.info(
            f'LINK_FORM: Processing additional question from {winning_chunk.processor_name}: "{additional_question}"'
        )

        result = func(self, chunks, winning_chunk, **input_kwargs)

        for chunk in chunks:
            logger.info(
                f'LINK_FORM: Evaluating {chunk.processor_name} with relevance {chunk.relevance}'
            )
            if chunk.relevance >= 0.6:
                logger.info(
                    f'LINK_FORM: Adding link between {winning_chunk.processor_name} and {chunk.processor_name} (relevance: {chunk.relevance})'
                )
            elif chunk.relevance <= 0.2:
                logger.info(
                    f'LINK_FORM: Removing link between {winning_chunk.processor_name} and {chunk.processor_name} (relevance: {chunk.relevance})'
                )

        return result

    return wrapper


def logging_fuse_processor(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, chunks: List[Any], query: str, **input_kwargs) -> Any:
        logger.info(
            f'FUSE_PROCESSOR: Starting fusion process with {len(chunks)} chunks'
        )

        result = func(self, chunks, query, **input_kwargs)

        logger.info('FUSE_PROCESSOR: Fusion process completed')
        return result

    return wrapper


def logging_processor_graph_operation(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, processor1_name: str, processor2_name: str) -> Any:
        operation_type = 'ADD' if func.__name__ == 'add_link' else 'REMOVE'
        logger.info(f'LINK {operation_type}: {processor1_name} <-> {processor2_name}')

        result = func(self, processor1_name, processor2_name)
        return result

    return wrapper


def logging_processor_neighbor_interaction(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, chunk: Any, query: str, **input_kwargs) -> Any:
        q = chunk.additional_question
        if q:
            logger.info(
                f'FUSE_PROCESSOR: Processing additional question from {chunk.processor_name}: "{q}"'
            )

        result = func(self, chunk, query, **input_kwargs)

        for nbr in self.processor_graph.get_neighbor_names(chunk.processor_name):
            if nbr == chunk.processor_name:  # self-link
                logger.info(
                    f'FUSE_PROCESSOR: Self-link detected for {nbr}, updating memory'
                )
            else:
                logger.info(
                    f'FUSE_PROCESSOR: Asking neighbor {nbr} for additional information'
                )
                logger.info(
                    f'FUSE_PROCESSOR: Added information from {nbr} to {chunk.processor_name}'
                )

        return result

    return wrapper
