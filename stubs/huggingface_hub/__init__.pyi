from typing import Any, Dict, Optional

class InferenceClient:
    def __init__(self, token: Optional[str] = None) -> None: ...
    def __call__(
        self,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
    def post(
        self,
        json: Dict[str, Any],
        model: str,
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str: ...
