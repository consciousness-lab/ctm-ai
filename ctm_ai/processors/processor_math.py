from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('math_processor')
class MathProcessor(BaseProcessor):
    REQUIRED_KEYS = ['OPENAI_API_KEY']

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger.create_messenger('math_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='math_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='hybrid_scorer')
