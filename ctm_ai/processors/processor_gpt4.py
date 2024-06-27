from typing import Any, Optional

from ..executors.executor_base import BaseExecutor
from ..messengers.messenger_base import BaseMessenger
from ..scorers.scorer_base import BaseScorer
from .processor_base import BaseProcessor


@BaseProcessor.register_processor('gpt4_processor')
class GPT4Processor(BaseProcessor):
    def __init__(
        self, name: str, group_name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(name, group_name, *args, **kwargs)

    def init_messenger(self) -> None:
        self.messenger = BaseMessenger(messenger_name='gpt4_messenger')

    def init_executor(self) -> None:
        self.executor = BaseExecutor(executor_name='gpt4_executor')

    def init_scorer(self) -> None:
        self.scorer = BaseScorer(scorer_name='gpt4_scorer')
