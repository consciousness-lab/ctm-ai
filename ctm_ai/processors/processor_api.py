from typing import Any, List

from openai import OpenAI

from .processor_base import BaseProcessor


@BaseProcessor.register_processor('api_processor')
class APIProcessor(BaseProcessor):
    REQUIRED_KEYS = ['OPENAI_API_KEY']

    def __init__(
        self,
        name: str,
        api_manager: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)

        self.api_manager = api_manager
        self.api_func_info = self.api_manager.funcs_to_all_info[name]
        self.client = OpenAI()


def register_api_processors(openai_function_names: List[str]):
    for openai_function_name in openai_function_names:
        processor_name = openai_function_name
        BaseProcessor._processor_registry[processor_name] = APIProcessor
