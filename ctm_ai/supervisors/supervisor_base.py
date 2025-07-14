from typing import Any, Dict, Optional, Tuple, Type, Union

from ..utils import configure_litellm, logging_ask


class BaseSupervisor(object):
    _supervisor_registry: Dict[str, Type['BaseSupervisor']] = {}

    @classmethod
    def register_supervisor(cls, name: str) -> Any:
        def decorator(
            subclass: Type['BaseSupervisor'],
        ) -> Type['BaseSupervisor']:
            cls._supervisor_registry[name] = subclass
            return subclass

        return decorator

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in cls._supervisor_registry:
            raise ValueError(f"No supervisor registered with name '{name}'")
        instance = super(BaseSupervisor, cls).__new__(cls._supervisor_registry[name])
        instance.name = name
        return instance

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.init_supervisor(*args, **kwargs)

    def init_supervisor(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the supervisor with LiteLLM support."""
        # Default configuration for LiteLLM
        self.model_name = kwargs.get('model', 'gpt-o3')
        self.info_model = kwargs.get('info_model', self.model_name)
        self.score_model = kwargs.get('score_model', self.model_name)
        self.max_tokens = kwargs.get('max_tokens', 300)
        self.temperature = kwargs.get('temperature', 0.0)

        # Configure LiteLLM settings
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Configure LiteLLM settings."""
        configure_litellm(model_name=self.model_name)

    @logging_ask()
    def ask(self, query: str, image_path: str) -> Tuple[Union[str, None], float]:
        gist = self.ask_info(query, image_path)
        score = self.ask_score(query, gist)
        return gist, score

    def ask_info(self, query: str, context: Optional[str] = None) -> str | None:
        raise NotImplementedError(
            "The 'ask_info' method must be implemented in derived classes."
        )

    def ask_score(self, query: str, gist: str | None) -> float:
        raise NotImplementedError(
            "The 'ask_score' method must be implemented in derived classes."
        )
