import base64
import io
from typing import Any, Dict, List, Optional

from ..utils import load_image
from .processor_base import BaseProcessor
from .processor_webagent_base import WebAgentBaseProcessor
from .prompts.webagent_prompts import (
    SCREENSHOT_SYSTEM_PROMPT,
    build_screenshot_user_prompt,
)


def _pil_to_base64(image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@BaseProcessor.register_processor("screenshot_processor")
class ScreenshotProcessor(WebAgentBaseProcessor):
    """Web-agent processor specialised in visual screenshot analysis."""

    def _build_web_prompt(
        self,
        objective: str,
        action_history: str,
        action_space: str,
        other_info: str,
        phase: str = "initial",
        **kwargs: Any,
    ) -> str:
        return build_screenshot_user_prompt(
            objective=objective,
            action_history=action_history,
            action_space=action_space,
            other_info=other_info,
            phase=phase,
        )

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        # Priority 1: pre-encoded base64 string routed by WebConsciousTuringMachine
        screenshot_b64 = getattr(self, "_screenshot_b64", None)
        image_path = kwargs.get("image_path")
        image = kwargs.get("image")

        if not screenshot_b64 and not image_path and image is None:
            return None

        if screenshot_b64:
            base64_image = screenshot_b64
        elif image_path:
            base64_image = load_image(image_path)
        else:
            base64_image = _pil_to_base64(image)

        system_prompt = self.system_prompt or SCREENSHOT_SYSTEM_PROMPT

        user_message: Dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
        return [
            {"role": "system", "content": system_prompt},
            user_message,
        ]
