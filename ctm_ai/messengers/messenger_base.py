from typing import Any, Callable, Dict, List, Optional, Type

from .message import Message


class BaseMessenger(object):
    _messenger_registry: Dict[str, Type['BaseMessenger']] = {}

    MESSENGER_CONFIGS = {
        'language_messenger': {
            'default_scorer_role': 'user',
            'format_query_with_prefix': True,
            'include_text_in_content': True,
        },
        'vision_messenger': {
            'default_scorer_role': 'assistant',
        },
        'audio_messenger': {
            'default_scorer_role': 'user',
        },
        'code_messenger': {
            'default_scorer_role': 'user',
            'format_query_with_prefix': True,
            'include_text_in_content': True,
        },
        'math_messenger': {
            'default_scorer_role': 'assistant',
            'include_query_in_scorer': False,
            'include_gists_in_scorer': False,
        },
        'search_messenger': {
            'default_scorer_role': 'assistant',
            'include_query_in_scorer': False,
            'include_gists_in_scorer': False,
        },
        'video_messenger': {
            'default_scorer_role': 'assistant',
            'format_query_with_prefix': True,
            'include_video_note': True,
        },
    }

    default_scorer_role: str = 'assistant'
    include_query_in_scorer: bool = True
    include_gists_in_scorer: bool = True

    default_executor_role: str = 'user'
    format_query_with_prefix: bool = False
    include_text_in_content: bool = False
    include_video_note: bool = False
    use_query_field: bool = False

    @classmethod
    def register_messenger(
        cls, name: str
    ) -> Callable[[Type['BaseMessenger']], Type['BaseMessenger']]:
        def decorator(
            subclass: Type['BaseMessenger'],
        ) -> Type['BaseMessenger']:
            cls._messenger_registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create_messenger(cls, name: str, *args: Any, **kwargs: Any) -> 'BaseMessenger':
        if name in cls._messenger_registry:
            return cls._messenger_registry[name](name, *args, **kwargs)
        elif name in cls.MESSENGER_CONFIGS:
            instance = object.__new__(cls)
            instance.name = name

            config = cls.MESSENGER_CONFIGS[name]
            for key, value in config.items():
                setattr(instance, key, value)

            instance.init_messenger(*args, **kwargs)
            return instance
        else:
            raise ValueError(
                f"No messenger registered or configured with name '{name}'"
            )

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> 'BaseMessenger':
        if name in cls._messenger_registry:
            instance = super(BaseMessenger, cls).__new__(cls._messenger_registry[name])
            instance.name = name
            return instance
        else:
            instance = super(BaseMessenger, cls).__new__(cls)
            instance.name = name
            return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_messenger(*args, **kwargs)

    def get_executor_messages(self) -> Any:
        return self.executor_messages

    def get_scorer_messages(self) -> Any:
        return self.scorer_messages

    def init_messenger(self) -> None:
        self.executor_messages: List[Message] = []
        self.scorer_messages: List[Message] = []

    def update(self, executor_output: Message, scorer_output: Message) -> None:
        self.executor_messages.append(executor_output)
        self.scorer_messages.append(scorer_output)

    def clear_memory(self) -> None:
        """Clear all stored messages"""
        self.executor_messages.clear()
        self.scorer_messages.clear()

    def _build_executor_content(
        self,
        query: str,
        text: Optional[str] = None,
        video_frames_path: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        content = query

        if self.format_query_with_prefix:
            content = f'Query: {query}\n'

        if self.include_text_in_content and text is not None:
            content += f'Text: {text}\n'

        if self.include_video_note and video_frames_path:
            content += f'Note: The input contains {len(video_frames_path)} video frames. Please integrate visual information across these frames for a comprehensive analysis.\n'

        # Add JSON format requirement
        content += """
You should utilize the other information in the context history and modality-specific information to answer the query.
Please respond in JSON format with the following structure:
{
    "response": "Your detailed response to the query",
    "additional_question": "If you are not sure about the answer, you should generate a question that potentially can be answered by other modality models or other tools like search engine."
}

Your additional_question should be potentially answerable by other modality models or other tools like search engine and about specific information that you are not sure about.
Your additional_question should be just about what kind of information you need to get from other modality models or other tools like search engine, nothing else about the task or original query should be included. For example, what is the tone of the audio, what is the facial expression of the person, what is the caption of the image, etc. The question needs to be short and clean."""

        return content

    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        image_path: Optional[str] = None,
        audio: Optional[Any] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[Any]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        memory_mode: bool = True,  # Default to memory mode
        **kwargs: Any,
    ) -> List[Message]:
        content = self._build_executor_content(
            query=query, text=text, video_frames_path=video_frames_path, **kwargs
        )

        message_data = {
            'role': self.default_executor_role,
        }

        if self.use_query_field:
            message_data['query'] = content
        else:
            message_data['content'] = content

        # Add multimodal information to message
        if image is not None:
            message_data['image'] = image
        if image_path is not None:
            message_data['image_path'] = image_path
        if audio is not None:
            message_data['audio'] = audio
        if audio_path is not None:
            message_data['audio_path'] = audio_path
        if video_frames is not None:
            message_data['video_frames'] = video_frames
        if video_frames_path is not None:
            message_data['video_frames_path'] = video_frames_path
        if video_path is not None:
            message_data['video_path'] = video_path

        message = Message(**message_data)

        # Always append to memory, but return different messages based on memory_mode
        self.executor_messages.append(message)
        if memory_mode:
            return self.executor_messages
        else:
            # Return only the current message without memory
            return [message]

    def collect_scorer_messages(
        self,
        executor_output: Message,
        query: str,
        memory_mode: bool = True,  # Default to memory mode
        **kwargs: Any,
    ) -> List[Message]:
        message_data = {
            'role': self.default_scorer_role,
            'gist': executor_output.gist,
        }

        if self.include_query_in_scorer:
            message_data['query'] = query

        if self.include_gists_in_scorer and hasattr(executor_output, 'gists'):
            message_data['gists'] = executor_output.gists

        message = Message(**message_data)

        # Always append to memory, but return different messages based on memory_mode
        self.scorer_messages.append(message)
        if memory_mode:
            return self.scorer_messages
        else:
            # Return only the current message without memory
            return [message]
