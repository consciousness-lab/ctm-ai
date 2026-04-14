import json
from typing import Any, Dict, Optional

DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    'relevance': 1.0,
    'confidence': 1.0,
    'surprise': 0.2,
}


class ConsciousTuringMachineConfig:
    DEFAULT_PARSE_PROMPT_TEMPLATE = """Based solely on the analysis provided below, give your final answer.

Your answer MUST start with either "Yes" or "No", followed by a brief explanation.

IMPORTANT: If the analysis expresses uncertainty, is inconclusive, or lacks sufficient evidence, you MUST answer "No".

Analysis:
{answer}
"""

    DEFAULT_FORCE_FINAL_PROMPT_TEMPLATE = (
        'Based on all analysis so far, produce the final answer.\n\n'
        'Analysis:\n{answer}\n'
    )

    def __init__(
        self,
        ctm_name: Optional[str] = None,
        max_iter_num: int = 3,
        output_threshold: float = 1.8,
        processors_config: Optional[Dict[str, Any]] = None,
        parse_model: str = 'gemini/gemini-2.5-flash-lite',
        parse_extra_body: Optional[Dict[str, Any]] = None,
        parse_temperature: float = 0.3,
        processor_temperature: float = 0.2,
        supervisors_model: str = 'gemini/gemini-2.5-flash-lite',
        supervisors_prompt: str = None,
        parse_prompt_template: Optional[str] = None,
        score_weights: Optional[Dict[str, float]] = None,
        num_additional_questions: int = 3,
        max_steps_before_force: int = 9,
        force_final_prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.ctm_name: Optional[str] = ctm_name
        self.max_iter_num: int = max_iter_num
        self.output_threshold: float = output_threshold
        self.processors_config: Dict[str, Any] = (
            processors_config if processors_config is not None else {}
        )
        self.parse_model = parse_model
        self.parse_extra_body: Optional[Dict[str, Any]] = parse_extra_body
        self.parse_temperature: float = parse_temperature
        self.processor_temperature: float = processor_temperature
        self.supervisors_model = supervisors_model
        self.supervisors_prompt = supervisors_prompt
        self.parse_prompt_template = (
            parse_prompt_template or self.DEFAULT_PARSE_PROMPT_TEMPLATE
        )
        self.max_steps_before_force: int = max_steps_before_force
        self.force_final_prompt_template = (
            force_final_prompt_template or self.DEFAULT_FORCE_FINAL_PROMPT_TEMPLATE
        )
        self.output_threshold = output_threshold
        self.score_weights: Dict[str, float] = {
            **DEFAULT_SCORE_WEIGHTS,
            **(score_weights or {}),
        }
        self.num_additional_questions: int = num_additional_questions
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
    def from_ctm(cls, ctm_name: Optional[str]) -> 'ConsciousTuringMachineConfig':
        if ctm_name is None:
            return cls()
        config_file = f'../ctm_conf/{ctm_name}_config.json'
        return cls.from_json_file(config_file)
