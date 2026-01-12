import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from litellm import completion
from numpy.typing import NDArray

from ..chunks import Chunk
from ..scorers import BaseScorer
from ..utils import configure_litellm, message_exponential_backoff
from .utils import JSON_FORMAT, parse_json_response


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

        self.model_name = kwargs.get('model', 'gemini/gemini-2.0-flash-lite')
        self.try_times = kwargs.get('try_times', 3)
        self.max_tokens = kwargs.get('max_tokens', 4096)
        self.return_num = kwargs.get('return_num', 1)
        self.temperature = kwargs.get('temperature', 0.2)
        self.fuse_history = []
        self.winner_answer = []
        self.all_context_history = []
        configure_litellm(model_name=self.model_name)

    def check_required_env_vars(self) -> None:
        missing_vars = [var for var in self.REQUIRED_KEYS if var not in os.environ]
        if missing_vars:
            raise EnvironmentError(
                f'[{self.name}] Missing required environment variables: {missing_vars}'
            )

    def add_fuse_history(self, question: str, answer: str) -> None:
        self.fuse_history.append({'additional_question': question, 'answer': answer})

    def add_all_context_history(
        self, query: str, answer: str, additional_question: str
    ) -> None:
        self.all_context_history.append(
            {
                'query': query,
                'answer': answer,
                'additional_question': additional_question,
            }
        )

    def _build_executor_content(
        self,
        query: str,
        text: Optional[str] = None,
        video_frames_path: Optional[List[str]] = None,
        is_fuse: bool = False,
        **kwargs: Any,
    ) -> str:
        content = query

        content = f'Query: {query}\n'

        if text is not None:
            content += f'Text: {text}\n'

        if video_frames_path:
            content += f'Note: The input contains {len(video_frames_path)} video frames. Please integrate visual information across these frames for a comprehensive analysis.\n'

        if not is_fuse:
            if len(self.fuse_history) > 0:
                content += '\nThere are extra information from other processors:\n'
                for i, item in enumerate(self.fuse_history, 1):
                    content += f'{i}. {item["answer"]}\n'

            if len(self.winner_answer) > 0:
                content += '\nThere are some previous answers to the same query, think further based on this answer:\n'
                for i, item in enumerate(self.winner_answer, 1):
                    content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        content += JSON_FORMAT
        return content

    @message_exponential_backoff()
    def ask_executor(
        self,
        messages: List[Dict[str, Any]],
        default_additional_question: str = '',
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, str]:
        response = completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            n=self.return_num,
            *args,
            **kwargs,
        )
        contents = [
            response.choices[i].message.content for i in range(len(response.choices))
        ]
        gist, additional_question = parse_json_response(
            contents[0], default_additional_question
        )
        return {
            'response': gist,
            'additional_question': additional_question,
        }

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError('Subclasses must implement this method')

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
        api_manager: Any = None,
        is_fuse: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        clean_query = query
        query = self._build_executor_content(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
            is_fuse=is_fuse,
        )
        executor_messages = self.build_executor_messages(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
            api_manager=api_manager,
        )
        executor_output = self.ask_executor(
            messages=executor_messages,
            default_additional_question='Would you like me to explain any specific aspects in more detail?',
        )
        if is_fuse:
            self.add_fuse_history(clean_query, executor_output['response'])
        self.add_all_context_history(
            clean_query,
            executor_output['response'],
            executor_output['additional_question'],
        )

        scorer = BaseScorer(*args, **kwargs)
        scorer_output = scorer.ask(query=query, messages=executor_output)
        additional_question = executor_output['additional_question'] or ''

        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )
        return chunk

    def get_memory_info(self) -> Tuple[int, int]:
        return {
            'all_history': self.all_context_history,
            'fuse_history': self.fuse_history,
            'winner_answer': self.winner_answer,
        }

    def update(self, chunk: Chunk) -> None:
        if chunk.processor_name != self.name:
            self.winner_answer.append(
                {'processor_name': chunk.processor_name, 'answer': chunk.gist}
            )

    def merge_outputs_into_chunk(
        self,
        name: str,
        executor_output: Dict[str, Any],
        scorer_output: Dict[str, float],
        additional_question: str = '',
    ) -> Chunk:
        return Chunk(
            time_step=0,
            processor_name=name,
            gist=executor_output['response'],
            relevance=scorer_output['relevance'],
            confidence=scorer_output['confidence'],
            surprise=scorer_output['surprise'],
            weight=scorer_output['weight'],
            additional_question=additional_question,
        )

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, BaseProcessor) and self.name == other.name
