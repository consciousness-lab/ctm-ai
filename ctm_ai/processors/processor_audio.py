from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('audio_processor')
class AudioProcessor(BaseProcessor):
    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger(name='audio_messenger')

    def init_executor(self) -> BaseExecutor:
        return BaseExecutor(name='audio_executor')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='language_scorer')
