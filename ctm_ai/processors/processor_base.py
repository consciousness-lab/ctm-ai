import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from litellm import completion
from numpy.typing import NDArray

from ..chunks import Chunk
from ..utils import (
    configure_litellm,
    get_completion_kwargs,
    get_model_provider,
    get_required_api_key_name,
    message_exponential_backoff,
)
from .utils import (
    JSON_FORMAT_FUSE,
    JSON_FORMAT_LINK_FORM,
    JSON_FORMAT_SCORE,
    build_json_format_score,
    parse_json_response_with_scores,
)


class BaseProcessor(object):
    """Base class for all processors.

    The executor LLM outputs answer, additional_question, and separate
    relevance, confidence, surprise scores in a single call. The final
    weight is computed as: relevance + confidence + (surprise × 0.2).
    """

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
        self.name = name
        self.group_name = group_name
        self.system_prompt = kwargs.get('system_prompt')
        self.model = kwargs.get('model') or 'gemini/gemini-2.5-flash-lite'
        self.model_name = kwargs.get('model') or 'gemini/gemini-2.5-flash-lite'

        # Provider-specific setup
        self.provider = get_model_provider(self.model)
        self._completion_kwargs = get_completion_kwargs(self.model)

        # Check env vars after model/provider is known
        self.check_required_env_vars()

        self.try_times = kwargs.get('try_times', 3)
        self.max_tokens = kwargs.get('max_tokens', 4096)
        self.return_num = kwargs.get('return_num', 1)
        self.temperature = kwargs.get('temperature', 0.2)
        self.fuse_history = []
        self.winner_answer = []
        self.all_context_history = []

        configure_litellm(model_name=self.model_name)

    def check_required_env_vars(self) -> None:
        # Separate provider API keys from other required keys
        provider_api_keys = {'GEMINI_API_KEY', 'DASHSCOPE_API_KEY', 'OPENAI_API_KEY'}
        non_provider_keys = [
            var for var in self.REQUIRED_KEYS if var not in provider_api_keys
        ]
        missing_vars = [var for var in non_provider_keys if var not in os.environ]

        # Check the provider-specific API key based on the configured model
        required_key = get_required_api_key_name(self.model)
        if required_key and required_key not in os.environ:
            missing_vars.append(required_key)

        if missing_vars:
            raise EnvironmentError(
                f'[{self.name}] Missing required environment variables: {missing_vars}'
            )

    def add_fuse_history(
        self, question: str, answer: str, processor_name: str = ''
    ) -> None:
        self.fuse_history.append(
            {
                'additional_question': question,
                'answer': answer,
                'processor_name': processor_name,
            }
        )

    def add_all_context_history(
        self, query: str, answer: str, additional_questions: List[str]
    ) -> None:
        self.all_context_history.append(
            {
                'query': query,
                'answer': answer,
                'additional_questions': additional_questions,
            }
        )

    def _build_executor_content(
        self,
        query: str,
        phase: str = 'initial',
        **kwargs: Any,
    ) -> str:
        content = f'Query: {query}\n'

        # Check if we have context history
        has_context = False

        # Add fuse_history for initial and link_form phases
        if phase in ('initial', 'link_form'):
            # Filter out empty answers from fuse_history
            valid_fuse_history = [
                item
                for item in self.fuse_history
                if item.get('answer') and str(item['answer']).strip()
            ]
            if len(valid_fuse_history) > 0:
                has_context = True
                content += '\n' + '=' * 60 + '\n'
                content += 'CONTEXT: Information from other modalities\n'
                content += '=' * 60 + '\n'
                for item in valid_fuse_history:
                    content += f'\n[{item["processor_name"]}]:\n{item["answer"]}\n'

        # Add winner_answer only for initial phase
        if phase == 'initial':
            # Filter out empty answers from winner_answer
            valid_winner_answer = [
                item
                for item in self.winner_answer
                if item.get('answer') and str(item['answer']).strip()
            ]
            if len(valid_winner_answer) > 0:
                has_context = True
                content += '\n' + '=' * 60 + '\n'
                content += 'CONTEXT: Previous answers to the same query\n'
                content += '=' * 60 + '\n'
                for item in valid_winner_answer:
                    content += f'\n[{item["processor_name"]}]:\n{item["answer"]}\n'

        # Add phase-specific format instructions
        if phase == 'initial':
            content += '\n' + '-' * 60 + '\n'
            content += 'INSTRUCTIONS\n'
            content += '-' * 60 + '\n'
            content += build_json_format_score(has_context=has_context)
        elif phase == 'link_form':
            content += '\n' + '-' * 60 + '\n'
            content += 'INSTRUCTIONS\n'
            content += '-' * 60 + '\n'
            content += JSON_FORMAT_LINK_FORM
        elif phase == 'fuse':
            content += JSON_FORMAT_FUSE

        return content

    @message_exponential_backoff()
    def ask_executor(
        self,
        messages: List[Dict[str, Any]],
        default_additional_questions: List[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = {
            **self._completion_kwargs,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'n': self.return_num,
            **kwargs,
        }
        response = completion(**call_kwargs)
        contents = [
            response.choices[i].message.content for i in range(len(response.choices))
        ]
        return parse_json_response_with_scores(
            contents[0], default_additional_questions
        )

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError('Subclasses must implement this method')

    @staticmethod
    def _extract_scores_from_executor_output(
        executor_output: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract relevance/confidence/surprise from executor output and compute weight."""
        relevance = float(executor_output.get('relevance', 0.5))
        confidence = float(executor_output.get('confidence', 0.5))
        surprise = float(executor_output.get('surprise', 0.5))
        return {
            'relevance': relevance,
            'confidence': confidence,
            'surprise': surprise,
            'weight': relevance + confidence + (surprise * 0.2),
        }

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
        phase: str = 'initial',
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        executor_content = self._build_executor_content(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
            phase=phase,
        )

        # Log the query content sent to this processor
        from ..utils import logger

        logger.info(
            f'\n{self.name} received query (phase={phase}):\n{executor_content[:500]}...'
            if len(executor_content) > 500
            else f'\n{self.name} received query (phase={phase}):\n{executor_content}'
        )

        executor_messages = self.build_executor_messages(
            query=executor_content,
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
        if executor_messages is None:
            return None

        executor_output = self.ask_executor(
            messages=executor_messages,
            default_additional_questions=(
                []
                if phase != 'initial'
                else [
                    'Would you like me to explain any specific aspects in more detail?'
                ]
            ),
        )

        # Handle different phases
        if phase == 'link_form':
            # Need response + relevance for link_form
            response = executor_output.get('response', '')
            relevance = float(executor_output.get('relevance', 0.5))
            return Chunk(
                time_step=0,
                processor_name=self.name,
                gist=response,
                relevance=relevance,
                confidence=0.0,
                surprise=0.0,
                weight=relevance,
                additional_questions=[],
                executor_content=executor_content,
            )
        elif phase == 'fuse':
            # Only need response for fuse
            response = executor_output.get('response')
            if response is None:
                return None
            return Chunk(
                time_step=0,
                processor_name=self.name,
                gist=response,
                relevance=0.0,
                confidence=0.0,
                surprise=0.0,
                weight=0.0,
                additional_questions=[],
                executor_content=executor_content,
            )

        # Initial phase - full processing
        if executor_output.get('response') is None:
            return None
        self.add_all_context_history(
            query,
            executor_output['response'],
            executor_output['additional_questions'],
        )

        # Extract scores from executor output
        scorer_output = self._extract_scores_from_executor_output(executor_output)
        additional_questions = executor_output['additional_questions'] or []

        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_questions=additional_questions,
            executor_content=executor_content,
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
        additional_questions: List[str] = None,
        executor_content: str = '',
    ) -> Chunk:
        return Chunk(
            time_step=0,
            processor_name=name,
            gist=executor_output['response'],
            relevance=scorer_output['relevance'],
            confidence=scorer_output['confidence'],
            surprise=scorer_output['surprise'],
            weight=scorer_output['weight'],
            additional_questions=additional_questions or [],
            executor_content=executor_content,
        )

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, BaseProcessor) and self.name == other.name
