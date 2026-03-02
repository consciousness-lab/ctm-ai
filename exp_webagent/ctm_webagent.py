import base64
import dataclasses
import io
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent, AgentInfo
from browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    overlay_som,
    prune_html,
)
from PIL import Image

from ctm_ai.ctms import WebCTM

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')

    with io.BytesIO() as buffer:
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # return f"data:image/jpeg;base64,{image_base64}"
    return image_base64


def _sanitize_send_msg_action(action: str) -> str:
    """Fix common LLM issues in send_msg_to_user actions.

    - Replace double curly braces {{ / }} with single { / }
      (LLMs sometimes mimic Python f-string escaping)
    """
    m = re.match(r'send_msg_to_user\((.+)\)\s*$', action, re.DOTALL)
    if not m:
        return action
    arg = m.group(1).strip()
    if arg.startswith(("'", '"')):
        quote = arg[0]
        inner = arg[1:]
        if inner.endswith(quote):
            inner = inner[:-1]
        inner = inner.replace('{{', '{').replace('}}', '}')
        return f'send_msg_to_user({quote}{inner}{quote})'
    return action


class CTMAgent(Agent):
    def __init__(
        self,
        ctm_name: str = 'web',
        use_html: bool = True,
        use_axtree: bool = True,
        use_screenshot: bool = True,
        demo_mode: str = 'off',
        task_log_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.ctm_name = ctm_name
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot

        self.ctm = WebCTM(ctm_name='web_ctm')

        self.action_set = HighLevelActionSet(
            subsets=['chat', 'tab', 'nav', 'bid', 'infeas'],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode,
        )

        self.action_history = []
        self.answer_history = []
        self.task_log_dir = task_log_dir
        self.step_count = 0

    def obs_preprocessor(self, obs: dict) -> dict:
        # Get extra_element_properties safely
        extra_element_properties = obs.get('extra_element_properties')

        # Create screenshot_som numpy array for saving (keep original format)
        screenshot_som_array = None
        screenshot_som_base64 = None

        if self.use_screenshot and extra_element_properties is not None:
            screenshot_som_array = overlay_som(
                screenshot=obs['screenshot'],
                extra_properties=extra_element_properties,
            )
            screenshot_som_base64 = image_to_jpg_base64_url(screenshot_som_array)

        return {
            'chat_messages': obs['chat_messages'],
            'screenshot': obs['screenshot'],
            'goal_object': obs['goal_object'],
            'last_action': obs['last_action'],
            'last_action_error': obs['last_action_error'],
            'open_pages_urls': obs['open_pages_urls'],
            'open_pages_titles': obs['open_pages_titles'],
            'active_page_index': obs['active_page_index'],
            'extra_element_properties': extra_element_properties,
            'axtree_txt': flatten_axtree_to_str(obs['axtree_object']),
            'pruned_html': prune_html(flatten_dom_to_str(obs['dom_object'])),
            # Keep original numpy array format for save_step_info
            'screenshot_som': screenshot_som_array,
            # Base64 version for CTM processing
            'screenshot_som_base64': screenshot_som_base64,
        }

    def get_action(self, obs: dict) -> tuple[str, AgentInfo]:
        input_params = self._prepare_ctm_inputs(obs)

        force_final = len(self.action_history) >= self.ctm.config.max_steps_before_force

        try:
            reasoning, action, parsed_answer = self.ctm(
                query=obs['goal_object'][0]['text'],
                action_space=input_params.get('action_space'),
                action_history=input_params.get('action_history'),
                html=obs['pruned_html'],
                axtree=obs['axtree_txt'],
                screenshot=obs['screenshot_som_base64'],
                other_info=input_params.get('other_info'),
                force_final=force_final,
            )

            if force_final and not action.strip().startswith('send_msg_to_user'):
                logger.info(
                    'Force final: overriding action with parsed_answer (step %d >= %d)',
                    len(self.action_history),
                    self.ctm.config.max_steps_before_force,
                )
                action = parsed_answer

            action = _sanitize_send_msg_action(action)

            self._save_step_log(obs)

            self.action_history.append(action)
            self.answer_history.append(parsed_answer)
            self.step_count += 1

            print('Reasoning: ', reasoning)
            print('Action:    ', action)

            return action, {}

        except Exception as e:
            logger.error(f'CTM error: {e}')
            fallback_action = (
                "send_msg_to_user('I encountered an error. Please try again.')"
            )
            self.action_history.append(fallback_action)
            self.step_count += 1

            info = AgentInfo(
                think=f'CTM error: {str(e)}',
                stats={'error': str(e)},
                extra_info={'fallback': True},
            )

            return fallback_action, info

    def _save_step_log(self, obs: dict) -> None:
        """Save the CTM step log (all iterations) to disk."""
        if not self.task_log_dir:
            return

        step_log = getattr(self.ctm, 'last_step_log', None)
        if step_log is None:
            return

        step_log['step'] = self.step_count

        active_idx = obs.get('active_page_index')
        urls = obs.get('open_pages_urls', [])
        if active_idx is not None and urls:
            idx = int(np.asarray(active_idx).flat[0])
            if 0 <= idx < len(urls):
                step_log['current_url'] = urls[idx]

        os.makedirs(self.task_log_dir, exist_ok=True)
        path = os.path.join(self.task_log_dir, f'step{self.step_count}.json')

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(step_log, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f'Step log saved to {path}')
        except Exception as exc:
            logger.warning(f'Failed to save step log: {exc}')

    def _prepare_ctm_inputs(self, obs: dict) -> Dict[str, Any]:
        inputs = {}

        inputs['other_info'] = 'Currently open tabs:\n'
        for page_index, (page_url, page_title) in enumerate(
            zip(obs['open_pages_urls'], obs['open_pages_titles'])
        ):
            inputs[
                'other_info'
            ] += f"""Tab {page_index}{' (active tab)' if page_index == obs['active_page_index'] else ''}
  Title: {page_title}
  URL: {page_url}
"""

        inputs['action_space'] = f"""\
# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the requested data. I will send it as structured JSON matching the required response schema.
```send_msg_to_user('{{"task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": ["Alice", "Bob"]}}')```

I have navigated to the requested page.
```send_msg_to_user('{{"task_type": "NAVIGATE", "status": "SUCCESS", "retrieved_data": null}}')```

CRITICAL: The argument to send_msg_to_user() MUST be a JSON string with keys "task_type", "status", and "retrieved_data", matching the response schema in the objective. Do NOT send plain text.

"""
        if self.action_history or self.answer_history:
            history_parts = []
            max_len = max(len(self.action_history), len(self.answer_history))
            for i in range(max_len):
                step_label = f'[Step {i}]'
                if i < len(self.action_history):
                    history_parts.append(
                        f'{step_label} Action: {self.action_history[i]}'
                    )
                if i < len(self.answer_history):
                    history_parts.append(
                        f'{step_label} Observation: {self.answer_history[i]}'
                    )
            inputs['action_history'] = '# History of past actions\n\n' + '\n\n'.join(
                history_parts
            )
        else:
            inputs['action_history'] = ''

        if obs['last_action_error']:
            inputs['action_history'] += f"""

# Error message from last action

{obs['last_action_error']}
"""

        return inputs

    def reset(self):
        """Reset the agent state."""
        self.action_history = []
        self.answer_history = []
        self.current_obs = None
        self.step_count = 0
        if hasattr(self.ctm, 'reset'):
            self.ctm.reset()


@dataclasses.dataclass
class CTMAgentArgs(AbstractAgentArgs):
    ctm_name: str = 'web'
    chat_mode: bool = False
    demo_mode: str = 'off'
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    task_log_dir: Optional[str] = None

    def make_agent(self):
        return CTMAgent(
            ctm_name=self.ctm_name,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            demo_mode=self.demo_mode,
            task_log_dir=self.task_log_dir,
        )
