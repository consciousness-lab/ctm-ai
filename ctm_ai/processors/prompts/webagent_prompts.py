import json
import re
from typing import Dict, List, Optional


AXTREE_SYSTEM_PROMPT = (
    "You are a UI Assistant specialized in interpreting web accessibility trees "
    "to control a browser. Your strength is understanding the semantic structure "
    "of web pages through accessibility information, identifying interactive "
    "elements by their bid (browser ID) values, and determining the most "
    "appropriate action based on the tree hierarchy and element roles."
)

HTML_SYSTEM_PROMPT = (
    "You are a UI Assistant specialized in analyzing raw HTML source code to "
    "control a browser. Your strength is understanding the DOM structure, "
    "locating elements by their attributes (id, class, name, type, aria-*), "
    "reading form fields and link targets, and determining appropriate actions "
    "from the HTML content."
)

SCREENSHOT_SYSTEM_PROMPT = (
    "You are a UI Assistant specialized in visual analysis of web page "
    "screenshots to control a browser. Your strength is understanding the "
    "visual layout, identifying UI elements by their appearance and position, "
    "reading visible text and labels, and interpreting the current interface "
    "state from a visual perspective."
)


_INSTRUCTIONS_BLOCK = """\
# Instructions
You are a UI Assistant helping a user perform tasks using a web browser. \
Review the task, the current page state, and all available information to \
determine the best next browser action. Think step by step, reflect on past \
actions and any errors, then produce your next best action.

# Critical Interaction Rules
Your answer will be interpreted and executed by a program — follow the \
formatting instructions exactly. Issue only ONE action at a time. Reflect on \
your past actions, any resulting error messages, and the current page state \
before deciding on your next action.\
"""

_INSTRUCTIONS_ANSWER_BLOCK = """\
# Instructions
You are a UI Assistant helping a user perform tasks using a web browser. \
Review the task, the current page state, and all available information to \
provide a concise answer based on this processor's modality. \
"""

_OUTPUT_RULES_BLOCK = """\
# Output Rules (read carefully)
1. Issue only one valid action at a time from the action space.
2. The `response` field must contain your internal reasoning and current \
progress analysis (e.g. "Last step I clicked Reports. Now I see the Shipping \
link with bid 123. I will click it now.").
3. The `action` field must contain a single valid command using `bid` values \
(numbers), e.g. click(\\"12\\"), fill('818', '456 Oak Avenue').
4. The `additional_question` should ONLY ask for missing perceptual information \
that another processor (screenshot, html, or axtree) could answer. Do NOT \
repeat the user's objective.
5. Use send_msg_to_user(...) ONLY to deliver the final answer to the user. \
If sufficient information is already visible, call it immediately.\
"""

_SCORE_RUBRIC_BLOCK = """\
# Self-Evaluation (evaluate your chosen action)

## Relevance (0.0 – 1.0): How well does your action advance the objective?
- 1.0 = Action directly and precisely progresses the task
- 0.8 = Action is clearly helpful, may not be the single optimal move
- 0.6 = Action is related but indirect or only partially helpful
- 0.4 = Weakly related; significant detour from the goal
- 0.0 = Wrong action or completely irrelevant to the objective

## Confidence (0.0 – 1.0): How certain are you the chosen action is correct?
- 1.0 = Very certain, clear evidence in the page data
- 0.8 = Mostly certain, minor ambiguity present
- 0.6 = Moderate certainty, some inference required
- 0.4 = Significant uncertainty, element identification is approximate
- 0.0 = Completely guessing, no reliable evidence available

## Surprise (0.0 – 1.0): How non-obvious is this action?
- 1.0 = Highly non-obvious, requires special reasoning or domain knowledge
- 0.6 = Mix of obvious and non-obvious elements
- 0.3 = Fairly predictable next step given the state
- 0.0 = Completely routine, expected next action\
"""

_JSON_INITIAL_FORMAT = """\
# JSON Output Format
{{
  "response": "Step-by-step reasoning about the current state and why you chose this action.",
  "action": "The single browser action to perform (e.g. click(\\"bid\\"), fill('bid', 'text')).",
  "additional_question": "A specific perceptual question for another processor to resolve uncertainty, or empty string if not needed.",
  "relevance": <number 0.0–1.0>,
  "confidence": <number 0.0–1.0>,
  "surprise": <number 0.0–1.0>
}}

Example 1 (navigation):
{{
  "response": "I clicked Reports last step. Now the sub-menu shows Shipping with bid 450. I will click it.",
  "action": "click(\\"450\\")",
  "additional_question": "In the screenshot, does a dark sub-menu appear with a Shipping link?",
  "relevance": 0.9,
  "confidence": 0.85,
  "surprise": 0.2
}}

Example 2 (final answer):
{{
  "response": "I found the requested information and will send it to the user.",
  "action": "send_msg_to_user(\\"The price for a 15\\\\" laptop is 1499 USD.\\")",
  "additional_question": "",
  "relevance": 1.0,
  "confidence": 0.95,
  "surprise": 0.1
}}\
"""

_JSON_LINK_FORM_FORMAT = """\
Answer the question(s) above using this processor's modality, then score how relevant your answer is to the question(s), can your response provide new information that is not already in the context that could help solve the problem?
(relevance: 1.0 = fully relevant, 0.0 = completely irrelevant).

# JSON Output Format
{{"response": "Your concrete answer to the question(s) from this modality's perspective.", "relevance": <number 0.0–1.0>}}
"""

_JSON_FUSE_FORMAT = """\
# JSON Output Format
{{
  "response": "Your synthesized answer integrating all available information. Do not include any browser action."
}}\
"""

# ---------------------------------------------------------------------------
# Per-processor additional-question hints (appended to the input section)
# ---------------------------------------------------------------------------

_AXTREE_ADDITIONAL_HINT = (
    "When uncertain about visual appearance (colors, exact positions, images), "
    "ask the screenshot processor. When uncertain about exact HTML attributes "
    "or non-visible metadata, ask the html processor."
)

_HTML_ADDITIONAL_HINT = (
    "When uncertain about the visual layout or which element is visually "
    "prominent, ask the screenshot processor. When uncertain about the exact "
    "bid value for a visually identified element, ask the axtree processor."
)

_SCREENSHOT_ADDITIONAL_HINT = (
    "When uncertain about the exact bid value for a visually identified "
    "element, ask the axtree processor. When uncertain about hidden fields "
    "or non-visible attributes, ask the html processor."
)

# ---------------------------------------------------------------------------
# Public prompt builders
# ---------------------------------------------------------------------------


def _format_other_info(other_info: str) -> str:
    return other_info if other_info and other_info.strip() else "None"


def build_axtree_user_prompt(
    objective: str,
    axtree: str,
    action_history: str,
    action_space: str,
    other_info: str,
    phase: str = "initial",
) -> str:
    """Build the user-turn prompt for the AXTree processor."""
    other_info_str = _format_other_info(other_info)
    if phase == "initial":
        base = (
            f"{_INSTRUCTIONS_BLOCK}\n\n"
            f"# Input Information\n"
            f"## User's objective\n{objective}\n\n"
            f"## Accessibility tree\n{axtree}\n\n"
            f"## Previous action\n{action_history}\n\n"
            f"## Action space\n{action_space}\n\n"
            f"## Additional info (outputs from other processors + history)\n"
            f"{other_info_str}\n\n"
            f"*Hint*: {_AXTREE_ADDITIONAL_HINT}\n\n"
        )
        return (
            base
            + _OUTPUT_RULES_BLOCK
            + "\n\n"
            + _SCORE_RUBRIC_BLOCK
            + "\n\n"
            + _JSON_INITIAL_FORMAT
        )
    base = (
        f"{_INSTRUCTIONS_ANSWER_BLOCK}\n\n"
        f"# Input Information\n"
        f"## User's objective\n{objective}\n\n"
        f"## Accessibility tree\n{axtree}\n\n"
        f"## Previous action\n{action_history}\n\n"
        f"## Additional info (outputs from other processors + history)\n"
        f"{other_info_str}\n\n"
        f"*Hint*: {_AXTREE_ADDITIONAL_HINT}\n\n"
    )
    if phase == "link_form":
        return base + _JSON_LINK_FORM_FORMAT
    # fuse
    return base + _JSON_FUSE_FORMAT


def build_html_user_prompt(
    objective: str,
    html: str,
    action_history: str,
    action_space: str,
    other_info: str,
    phase: str = "initial",
) -> str:
    """Build the user-turn prompt for the HTML processor."""
    other_info_str = _format_other_info(other_info)
    if phase == "initial":
        base = (
            f"{_INSTRUCTIONS_BLOCK}\n\n"
            f"# Input Information\n"
            f"## User's objective\n{objective}\n\n"
            f"## HTML source\n{html}\n\n"
            f"## Previous action\n{action_history}\n\n"
            f"## Action space\n{action_space}\n\n"
            f"## Additional info (outputs from other processors + history)\n"
            f"{other_info_str}\n\n"
            f"*Hint*: {_HTML_ADDITIONAL_HINT}\n\n"
        )
        return (
            base
            + _OUTPUT_RULES_BLOCK
            + "\n\n"
            + _SCORE_RUBRIC_BLOCK
            + "\n\n"
            + _JSON_INITIAL_FORMAT
        )
    base = (
        f"{_INSTRUCTIONS_ANSWER_BLOCK}\n\n"
        f"# Input Information\n"
        f"## User's objective\n{objective}\n\n"
        f"## HTML source\n{html}\n\n"
        f"## Previous action\n{action_history}\n\n"
        f"## Additional info (outputs from other processors + history)\n"
        f"{other_info_str}\n\n"
        f"*Hint*: {_HTML_ADDITIONAL_HINT}\n\n"
    )
    if phase == "link_form":
        return base + _JSON_LINK_FORM_FORMAT
    return base + _JSON_FUSE_FORMAT


def build_screenshot_user_prompt(
    objective: str,
    action_history: str,
    action_space: str,
    other_info: str,
    phase: str = "initial",
) -> str:
    """Build the text portion of the user-turn prompt for the Screenshot processor.

    The actual screenshot image is attached separately in the message payload.
    """
    other_info_str = _format_other_info(other_info)
    if phase == "initial":
        base = (
            f"{_INSTRUCTIONS_BLOCK}\n\n"
            f"# Input Information\n"
            f"## User's objective\n{objective}\n\n"
            f"## Screenshot\n[See the attached screenshot image.]\n\n"
            f"## Previous action\n{action_history}\n\n"
            f"## Action space\n{action_space}\n\n"
            f"## Additional info (outputs from other processors + history)\n"
            f"{other_info_str}\n\n"
            f"*Hint*: {_SCREENSHOT_ADDITIONAL_HINT}\n\n"
        )
        return (
            base
            + _OUTPUT_RULES_BLOCK
            + "\n\n"
            + _SCORE_RUBRIC_BLOCK
            + "\n\n"
            + _JSON_INITIAL_FORMAT
        )
    base = (
        f"{_INSTRUCTIONS_ANSWER_BLOCK}\n\n"
        f"# Input Information\n"
        f"## User's objective\n{objective}\n\n"
        f"## Screenshot\n[See the attached screenshot image.]\n\n"
        f"## Previous action\n{action_history}\n\n"
        f"## Additional info (outputs from other processors + history)\n"
        f"{other_info_str}\n\n"
        f"*Hint*: {_SCREENSHOT_ADDITIONAL_HINT}\n\n"
    )
    if phase == "link_form":
        return base + _JSON_LINK_FORM_FORMAT
    return base + _JSON_FUSE_FORMAT


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _extract_json_safe(content: str) -> dict:
    """Attempt to extract a JSON object from raw LLM output."""
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.rfind("```")
        if start > 6 and end > start:
            try:
                return json.loads(content[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass
    # Regex fallback
    parsed: Dict = {}
    for key in ("relevance", "confidence", "surprise"):
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
        if m:
            try:
                parsed[key] = float(m.group(1))
            except ValueError:
                pass
    for key in ("response", "action", "additional_question"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
        if m:
            parsed[key] = m.group(1).replace('\\"', '"')
    return parsed


def parse_webagent_response(
    content: str,
    default_additional_questions: Optional[List[str]] = None,
) -> Dict:
    if default_additional_questions is None:
        default_additional_questions = []

    def _safe_float(val, lo=0.0, hi=1.0, default=0.5) -> float:
        try:
            return max(lo, min(hi, float(val)))
        except (ValueError, TypeError):
            return default

    parsed = _extract_json_safe(content)

    action = parsed.get("action", "").strip()
    reasoning = parsed.get("response", content).strip()

    # The ``additional_question`` from web agent maps to additional_questions
    aq = parsed.get("additional_question", "").strip()
    additional_questions = [aq] if aq else list(default_additional_questions)

    return {
        # action is the primary output (goes into Chunk.gist)
        "response": action if action else reasoning,
        "reasoning": reasoning,
        "additional_questions": additional_questions,
        "relevance": _safe_float(parsed.get("relevance", 0.5)),
        "confidence": _safe_float(parsed.get("confidence", 0.5)),
        "surprise": _safe_float(parsed.get("surprise", 0.2)),
    }
