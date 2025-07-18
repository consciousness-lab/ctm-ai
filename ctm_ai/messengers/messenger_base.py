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

        message = Message(**message_data)
        self.executor_messages.append(message)
        return self.executor_messages

    def collect_scorer_messages(
        self, executor_output: Message, query: str, **kwargs: Any
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
        self.scorer_messages.append(message)
        return self.scorer_messages
