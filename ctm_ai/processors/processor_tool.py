from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('tool_processor')
class ToolProcessor(BaseProcessor):
    REQUIRED_KEYS = ['OPENAI_API_KEY', 'TOOLBENCH_KEY']

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='tool_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='tool_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='tool_scorer')
