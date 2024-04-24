from typing import Any, Dict, Optional

class InferenceClient:
    def __init__(self, model_id: str, token: Optional[str] = None) -> None: ...
    def __call__(
        self,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
