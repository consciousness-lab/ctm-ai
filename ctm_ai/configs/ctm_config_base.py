import json
from typing import Any, Dict, Optional


class BaseConsciousnessTuringMachineConfig:
    def __init__(
        self,
        ctm_name: Optional[str] = None,
        max_iter_num: int = 3,
        output_threshold: float = 0.5,
        groups_of_processors: Dict[
            str, Any
        ] = {},  # Better to avoid mutable default arguments
        scorer: str = 'gpt4_scorer',
        supervisor: str = 'gpt4_supervisor',
        **kwargs: Any,
    ) -> None:
        self.ctm_name: Optional[str] = ctm_name
        self.max_iter_num: int = max_iter_num
        self.output_threshold: float = output_threshold
        self.groups_of_processors: Dict[str, Any] = groups_of_processors
        self.scorer: str = scorer
        self.supervisor: str = supervisor
        # Handle additional, possibly unknown configuration parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.__dict__, indent=2) + '\n'

    @classmethod
    def from_json_file(cls, json_file: str) -> 'BaseConsciousnessTuringMachineConfig':
        """Creates an instance from a JSON file."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls(**json.loads(text))

    @classmethod
    def from_ctm(cls, ctm_name: str) -> 'BaseConsciousnessTuringMachineConfig':
        """
        Simulate fetching a model configuration from a ctm model repository.
        This example assumes the configuration is already downloaded and saved locally.
        """
        # This path would be generated dynamically based on `model_name_or_path`
        # For simplicity, we're directly using it as a path to a local file
        config_file = f'../ctm_conf/{ctm_name}_config.json'
        return cls.from_json_file(config_file)
