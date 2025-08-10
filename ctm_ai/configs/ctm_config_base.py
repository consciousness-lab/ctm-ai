import json
from typing import Any, Dict, Optional


class ConsciousTuringMachineConfig:
    def __init__(
        self,
        ctm_name: Optional[str] = None,
        max_iter_num: int = 3,
        output_threshold: float = 0.5,
        scorer: str = 'api_scorer',
        scorer_use_llm: bool = True,
        supervisor: str = 'language_supervisor',
        processors_config: Optional[Dict[str, Any]] = None,
        scorer_model: str = 'gemini/gemini-2.0-flash-lite',
        supervisors_model: str = 'gemini/gemini-2.0-flash-lite',
        **kwargs: Any,
    ) -> None:
        self.ctm_name: Optional[str] = ctm_name
        self.max_iter_num: int = max_iter_num
        self.output_threshold: float = output_threshold
        self.scorer: str = scorer
        self.scorer_use_llm: bool = scorer_use_llm
        self.supervisor: str = supervisor
        self.processors_config: Dict[str, Any] = (
            processors_config if processors_config is not None else {}
        )
        self.scorer_model = scorer_model
        self.supervisors_model = supervisors_model
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=2) + '\n'

    @classmethod
    def from_json_file(cls, json_file: str) -> 'ConsciousTuringMachineConfig':
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls(**json.loads(text))

    @classmethod
    def from_ctm(cls, ctm_name: str) -> 'ConsciousTuringMachineConfig':
        config_file = f'../ctm_conf/{ctm_name}_config.json'
        return cls.from_json_file(config_file)
