from typing import List

from ..apis import BaseEnv
from ..chunks import Chunk
from ..executors import BaseExecutor
from ..messengers import BaseMessenger
from ..scorers import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('tool_processor')
class ToolProcessor(BaseProcessor):
    REQUIRED_KEYS = ['OPENAI_API_KEY', 'TOOLBENCH_KEY']

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger.create_messenger('tool_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='tool_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='language_scorer')

    def ask(
        self,
        query: str,
        io_function: BaseEnv,
        openai_function_name: str,
    ) -> Chunk:
        executor_messages = self.messenger.collect_executor_messages(
            query=query,
            io_function=io_function,
            openai_function_name=openai_function_name,
        )

        executor_output = self.executor.ask(
            messages=executor_messages,
            io_function=io_function,
            openai_function_name=openai_function_name,
        )

        scorer_messages = self.messenger.collect_scorer_messages(
            query=query,
            io_function=io_function,
            openai_function_name=openai_function_name,
            executor_output=executor_output,
        )

        scorer_output = self.scorer.ask(messages=scorer_messages)

        self.messenger.update(
            executor_output=executor_output,
            scorer_output=scorer_output,
        )

        chunk = self.merge_outputs_into_chunk(
            name=self.name, scorer_output=scorer_output, executor_output=executor_output
        )
        return chunk


def register_tool_processors(openai_function_names: List[str]):
    for openai_function_name in openai_function_names:
        processor_name = openai_function_name
        BaseProcessor._processor_registry[processor_name] = ToolProcessor
