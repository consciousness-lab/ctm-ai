from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('search_processor')
class SearchProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GOOGLE_API_KEY', 'GOOGLE_CSE_ID']

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='search_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='search_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='hybrid_scorer')
