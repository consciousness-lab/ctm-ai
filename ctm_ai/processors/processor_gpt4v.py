from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('gpt4v_processor')
class GPT4VProcessor(BaseProcessor):
    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='gpt4_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='gpt4v_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='gpt4_scorer')
