from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('audio_processor')
class AudioProcessor(BaseProcessor):
    REQUIRED_KEYS = ['GEMINI_API_KEY']

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger.create_messenger('audio_messenger')

    def init_executor(
        self, system_prompt: str = None, model: str = None
    ) -> BaseExecutor:
        return BaseExecutor(
            name='audio_executor', system_prompt=system_prompt, model=model
        )

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='language_scorer')
