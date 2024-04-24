# inference_api.pyi

from typing import Any, Dict

class InferenceApi:
    def __init__(self, model_id: str, api_key: str) -> None: ...
    def query(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...
