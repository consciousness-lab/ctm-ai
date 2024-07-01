from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('wolfram_alpha_processor')
class WolframAlphaProcessor(BaseProcessor):
    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='wolfram_alpha_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='wolfram_alpha_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='gpt4_scorer')
