from typing import Any, List

from ..chunks import Chunk
from ..executors import BaseExecutor
from ..messengers import BaseMessenger
from ..scorers import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('api_processor')
class APIProcessor(BaseProcessor):
    REQUIRED_KEYS = ['OPENAI_API_KEY']

    def __init__(self, name: str, group_name: str = None, *args, **kwargs):
        # Extract api_manager from kwargs before calling super().__init__
        self.api_manager = kwargs.get('api_manager', None)
        super().__init__(name, group_name, *args, **kwargs)

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger.create_messenger('api_messenger')

    def init_executor(
        self, system_prompt: str = None, model: str = None
    ) -> BaseExecutor:
        return BaseExecutor(
            name='api_executor',
            system_prompt=system_prompt,
            model=model,
            api_manager=self.api_manager,
            function_name=self.name,  # Pass processor name as function name
        )

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='api_scorer')

    def ask(
        self,
        query: str,
        api_manager: Any = None,
        use_memory: bool = True,  # Whether to condition on memory
        store_memory: bool = True,  # Whether to store input-output pair in memory
        **kwargs,
    ) -> Chunk:
        # Collect executor messages with or without memory
        executor_messages = self.messenger.collect_executor_messages(
            query=query,
            api_manager=api_manager,
            use_memory=use_memory,
            store_memory=store_memory,
        )

        # Ask executor
        executor_output = self.executor.ask(
            messages=executor_messages,
            api_manager=api_manager,
        )

        # Collect scorer messages with or without memory
        scorer_messages = self.messenger.collect_scorer_messages(
            query=query,
            executor_output=executor_output,
            api_manager=api_manager,
            use_memory=use_memory,
            store_memory=store_memory,
        )
        scorer_use_llm = (
            getattr(self.config, 'scorer_use_llm', True)
            if hasattr(self, 'config')
            else True
        )
        scorer_output = self.scorer.ask(
            messages=scorer_messages, use_llm=scorer_use_llm
        )

        # Store in memory if specified
        if store_memory:
            self.messenger.update(
                executor_output=executor_output,
                scorer_output=scorer_output,
            )

        # Use additional_question from executor output
        additional_question = executor_output.additional_question or ''

        # Merge outputs into a chunk
        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )
        return chunk


def register_api_processors(openai_function_names: List[str]):
    for openai_function_name in openai_function_names:
        processor_name = openai_function_name
        BaseProcessor._processor_registry[processor_name] = APIProcessor
