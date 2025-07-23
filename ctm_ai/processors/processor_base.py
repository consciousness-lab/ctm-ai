import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk
from ..executors import BaseExecutor
from ..messengers import BaseMessenger, Message
from ..scorers import BaseScorer


class BaseProcessor(object):
    _processor_registry: Dict[str, Type['BaseProcessor']] = {}
    REQUIRED_KEYS: List[str] = []

    @classmethod
    def register_processor(
        cls, name: str
    ) -> Callable[[Type['BaseProcessor']], Type['BaseProcessor']]:
        def decorator(
            subclass: Type['BaseProcessor'],
        ) -> Type['BaseProcessor']:
            cls._processor_registry[name] = subclass
            return subclass

        return decorator

    def __new__(
        cls,
        name: str,
        group_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> 'BaseProcessor':
        if name not in cls._processor_registry:
            raise ValueError(f"No processor registered with name '{name}'")
        subclass = cls._processor_registry[name]
        instance = super(BaseProcessor, cls).__new__(subclass)
        instance.name = name
        instance.group_name = group_name
        return instance

    def __init__(
        self, name: str, group_name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        self.check_required_env_vars()
        self.name = name
        self.group_name = group_name
        self.system_prompt = kwargs.get('system_prompt')
        self.model = kwargs.get('model')

        self.executor = self.init_executor(
            system_prompt=self.system_prompt, model=self.model
        )
        self.messenger = self.init_messenger()
        self.scorer = self.init_scorer()

        # Set the system prompt in the messenger after initialization
        if self.system_prompt:
            self.messenger.system_prompt_message = Message(
                role='system', content=self.system_prompt
            )

    def check_required_env_vars(self) -> None:
        missing_vars = [var for var in self.REQUIRED_KEYS if var not in os.environ]
        if missing_vars:
            raise EnvironmentError(
                f'[{self.name}] Missing required environment variables: {missing_vars}'
            )

    def init_executor(
        self, system_prompt: Optional[str] = None, model: Optional[str] = None
    ) -> BaseExecutor:
        return BaseExecutor(
            name='language_executor', system_prompt=system_prompt, model=model
        )

    def init_messenger(self) -> BaseMessenger:
        return BaseMessenger.create_messenger('language_messenger')

    def init_scorer(self) -> BaseScorer:
        return BaseScorer(name='language_scorer')

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        use_memory: bool = True,  # Whether to condition on memory
        store_memory: bool = True,  # Whether to store input-output pair in memory
    ) -> Chunk:
        # Collect executor messages with or without memory
        executor_messages = self.messenger.collect_executor_messages(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
            use_memory=use_memory,
            store_memory=store_memory,
        )

        # Ask executor
        executor_output = self.executor.ask(
            messages=executor_messages,
            image_path=image_path,
            audio_path=audio_path,
            video_frames_path=video_frames_path,
            video_path=video_path,
        )

        # Collect scorer messages with or without memory
        scorer_messages = self.messenger.collect_scorer_messages(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
            executor_output=executor_output,
            use_memory=use_memory,
            store_memory=store_memory,
        )

        # Ask scorer
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

    def clear_memory(self) -> None:
        """Clear all stored messages in messenger"""
        self.messenger.clear_memory()

    def get_memory_size(self) -> Tuple[int, int]:
        """Get the number of stored executor and scorer messages"""
        return (
            len(self.messenger.executor_messages),
            len(self.messenger.scorer_messages),
        )

    def get_memory_content(self) -> Dict[str, List[Message]]:
        """Get the content of processor memory"""
        return {
            'executor_messages': self.messenger.executor_messages.copy(),
            'scorer_messages': self.messenger.scorer_messages.copy(),
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of processor memory"""
        executor_count = len(self.messenger.executor_messages)
        scorer_count = len(self.messenger.scorer_messages)

        # Get recent messages for summary
        recent_executor = (
            self.messenger.executor_messages[-1] if executor_count > 0 else None
        )
        recent_scorer = self.messenger.scorer_messages[-1] if scorer_count > 0 else None

        return {
            'processor_name': self.name,
            'memory_mode': self.memory_mode,
            'executor_message_count': executor_count,
            'scorer_message_count': scorer_count,
            'recent_executor_gist': recent_executor.gist if recent_executor else None,
            'recent_scorer_scores': {
                'relevance': recent_scorer.relevance if recent_scorer else None,
                'confidence': recent_scorer.confidence if recent_scorer else None,
                'surprise': recent_scorer.surprise if recent_scorer else None,
                'weight': recent_scorer.weight if recent_scorer else None,
            }
            if recent_scorer
            else None,
        }

    def update(self, chunk: Chunk) -> None:
        if chunk.processor_name != self.name:
            executor_output, scorer_output = self.split_chunk_into_outputs(chunk)
            self.messenger.update(
                executor_output=executor_output,
                scorer_output=scorer_output,
            )

    def merge_outputs_into_chunk(
        self,
        name: str,
        scorer_output: Message,
        executor_output: Message,
        additional_question: str = '',
    ) -> Chunk:
        return Chunk(
            time_step=0,
            processor_name=name,
            gist=executor_output.gist,
            relevance=scorer_output.relevance,
            confidence=scorer_output.confidence,
            surprise=scorer_output.surprise,
            weight=scorer_output.weight,
            intensity=scorer_output.weight,
            mood=scorer_output.weight,
            additional_question=additional_question,
        )

    def split_chunk_into_outputs(self, chunk: Chunk) -> Tuple[Message, Message]:
        executor_output = Message(
            role='assistant',
            gist=chunk.gist,
        )
        scorer_output = Message(
            relevance=chunk.relevance,
            confidence=chunk.confidence,
            surprise=chunk.surprise,
            weight=chunk.weight,
        )
        return executor_output, scorer_output

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, BaseProcessor) and self.name == other.name
