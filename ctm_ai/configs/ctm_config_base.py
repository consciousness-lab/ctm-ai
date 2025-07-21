import json
from typing import Any, Dict, Optional


class ConsciousnessTuringMachineConfig:
    def __init__(
        self,
        ctm_name: Optional[str] = None,
        max_iter_num: int = 3,
        output_threshold: float = 0.5,
        groups_of_processors: Optional[Dict[str, Any]] = None,
        scorer: str = 'language_scorer',
        supervisor: str = 'language_supervisor',
        fuser: str = 'language_fuser',
        redundant_text_sim_threshold: float = 0.8,
        redundant_weight_threshold: float = 0.5,
        synergy_text_sim_threshold: float = 0.2,
        synergy_weight_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.ctm_name: Optional[str] = ctm_name
        self.max_iter_num: int = max_iter_num
        self.output_threshold: float = output_threshold
        self.groups_of_processors: Dict[str, Any] = (
            groups_of_processors if groups_of_processors is not None else {}
        )
        self.scorer: str = scorer
        self.supervisor: str = supervisor
        self.fuser: str = fuser
        self.redundant_text_sim_threshold: float = redundant_text_sim_threshold
        self.redundant_weight_threshold: float = redundant_weight_threshold
        self.synergy_text_sim_threshold: float = synergy_text_sim_threshold
        self.synergy_weight_threshold: float = synergy_weight_threshold
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=2) + '\n'

    @classmethod
    def from_json_file(cls, json_file: str) -> 'ConsciousnessTuringMachineConfig':
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls(**json.loads(text))

    @classmethod
    def from_ctm(cls, ctm_name: str) -> 'ConsciousnessTuringMachineConfig':
        config_file = f'../ctm_conf/{ctm_name}_config.json'
        return cls.from_json_file(config_file)
