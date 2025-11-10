import base64
import dataclasses
import io
import logging
from typing import Dict, Any

import numpy as np
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent, AgentInfo
from browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    prune_html,
    overlay_som,
)
from ctm_ai.ctms.ctm_webagent import WebConsciousTuringMachine

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class CTMAgent(Agent):

    def __init__(
        self,
        ctm_name: str = "web",
        use_html: bool = True,
        use_axtree: bool = True,
        use_screenshot: bool = True,
        demo_mode: str = "off",
    ) -> None:
        super().__init__()

        self.ctm_name = ctm_name
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot

        self.ctm = WebConsciousTuringMachine(ctm_name=ctm_name)

        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode,
        )

        self.action_history = []
        self.answer_history = []

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
            "screenshot_som": (
                image_to_jpg_base64_url(
                    overlay_som(
                        screenshot=obs["screenshot"],
                        extra_properties=obs["extra_element_properties"],
                    )
                )
                if self.use_screenshot
                and obs.get("extra_element_properties", None) is not None
                else None
            ),
        }

    def get_action(self, obs: dict) -> tuple[str, AgentInfo]:
        input_params = self._prepare_ctm_inputs(obs)
        breakpoint()

        try:
            answer, action = self.ctm(
                query=obs["goal_object"][0]["text"],
                action_space=input_params.get("action_space"),
                action_history=input_params.get("action_history"),
                html=obs["pruned_html"],
                axtree=obs["axtree_txt"],
                screenshot=obs["screenshot_som"],
                other_info=input_params.get("other_info"),
            )

            self.action_history.append(action)
            self.answer_history.append(answer)

            print("Answer: ", answer)
            print("Action: ", action)

            return action, {}

        except Exception as e:
            logger.error(f"CTM error: {e}")
            fallback_action = (
                "send_msg_to_user('I encountered an error. Please try again.')"
            )
            self.action_history.append(fallback_action)

            info = AgentInfo(
                think=f"CTM error: {str(e)}",
                stats={"error": str(e)},
                extra_info={"fallback": True},
            )

            return fallback_action, info

    def _prepare_ctm_inputs(self, obs: dict) -> Dict[str, Any]:
        inputs = {}

        inputs["other_info"] = "Currently open tabs:\n"
        for page_index, (page_url, page_title) in enumerate(
            zip(obs["open_pages_urls"], obs["open_pages_titles"])
        ):
            inputs[
                "other_info"
            ] += f"""Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
  Title: {page_title}
  URL: {page_url}
"""

        inputs[
            "action_space"
        ] = f"""\
# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

"""
        if self.action_history or self.answer_history:
            history_parts = []
            max_len = max(len(self.action_history), len(self.answer_history))
            for i in range(max_len):
                if i < len(self.answer_history):
                    history_parts.append(f"Answer: {self.answer_history[i]}")
                if i < len(self.action_history):
                    history_parts.append(f"Action: {self.action_history[i]}")
            inputs["action_history"] = "# History of past actions\n\n" + "\n\n".join(
                history_parts
            )
        else:
            inputs["action_history"] = ""

        if obs["last_action_error"]:
            inputs[
                "action_history"
            ] += f"""

# Error message from last action

{obs['last_action_error']}
"""

        return inputs

    def reset(self):
        """Reset the agent state."""
        self.action_history = []
        self.answer_history = []
        self.current_obs = None
        if hasattr(self.ctm, "reset"):
            self.ctm.reset()


@dataclasses.dataclass
class CTMAgentArgs(AbstractAgentArgs):

    ctm_name: str = "web"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True

    def make_agent(self):
        return CTMAgent(
            ctm_name=self.ctm_name,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            demo_mode=self.demo_mode,
        )
