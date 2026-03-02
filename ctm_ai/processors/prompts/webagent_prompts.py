import json
import re
from typing import Dict, List, Optional


AXTREE_SYSTEM_PROMPT = (
    "You are a UI Assistant specialized in interpreting web accessibility trees "
    "to control a browser. Your strength is understanding the semantic structure "
    "of web pages through accessibility information, identifying interactive "
    "elements by their bid (browser ID) values, and determining the most "
    "appropriate action based on the tree hierarchy and element roles. "
    "IMPORTANT: You MUST always respond with a single valid JSON object exactly "
    "matching the requested format. Do NOT output plain text, markdown prose, "
    "or any content outside the JSON object."
)

HTML_SYSTEM_PROMPT = (
    "You are a UI Assistant specialized in analyzing raw HTML source code to "
    "control a browser. Your strength is understanding the DOM structure, "
    "locating elements by their attributes (id, class, name, type, aria-*), "
    "reading form fields and link targets, and determining appropriate actions "
    "from the HTML content. "
    "IMPORTANT: You MUST always respond with a single valid JSON object exactly "
    "matching the requested format. Do NOT output plain text, markdown prose, "
    "or any content outside the JSON object."
)

SCREENSHOT_SYSTEM_PROMPT = (
    "You are a UI Assistant specialized in visual analysis of web page "
    "screenshots to control a browser. Your strength is understanding the "
    "visual layout, identifying UI elements by their appearance and position, "
    "reading visible text and labels, and interpreting the current interface "
    "state from a visual perspective. "
    "IMPORTANT: You MUST always respond with a single valid JSON object exactly "
    "matching the requested format. Do NOT output plain text, markdown prose, "
    "or any content outside the JSON object."
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
before deciding on your next action.

# MANDATORY OUTPUT FORMAT
You MUST respond with a single valid JSON object. \
Do NOT include any text, explanation, or markdown outside the JSON. \
Do NOT use plain-text labeled fields like "Reasoning: ..." or "Action: ...". \
Your entire response must be parseable by json.loads().\
"""

_INSTRUCTIONS_ANSWER_BLOCK = """\
# Instructions
You are a UI Assistant helping a user perform tasks using a web browser. \
Review the task, the current page state, and all available information to \
provide a concise answer based on this processor's modality. \
Do NOT suggest or output any browser action.

# MANDATORY OUTPUT FORMAT
You MUST respond with a single valid JSON object. \
Do NOT include any text, explanation, or markdown outside the JSON. \
Do NOT use plain-text labeled fields like "Response: ..." or "Relevance: ...". \
Your entire response must be parseable by json.loads().\
"""

_OUTPUT_RULES_BLOCK = """\
# Output Rules (read carefully)
1. Your ENTIRE response must be a single valid JSON object — no text before or after it.
2. Issue only one valid action at a time from the action space.
3. The `response` field MUST include ALL of the following:
   a) What you observe on the current page (key elements, counts, scroll state).
   b) Task progress: what has been done so far vs. what remains.
   c) Why you chose this specific action as the next step.
4. The `action` field must contain a single valid command using `bid` values \
(numbers), e.g. click(\\"12\\"), fill('818', '456 Oak Avenue').
5. Use `additional_question` Your additional_questions should be potentially answerable by other processors(screenshot, html, axtree) about specific information that you are not sure about, nothing else about the task or original query should be included.

6. BEFORE calling send_msg_to_user(...), verify in your reasoning that you \
have gathered ALL required information. If there is unseen content (more \
pages, scroll, pagination, unexplored sections), continue exploring first.
7. The argument to send_msg_to_user() MUST be a JSON string matching the \
response schema provided in the objective. It must contain "task_type", \
"status", and "retrieved_data" fields. Do NOT send plain natural language text.\
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

## Surprise (0.0 – 1.0): Is your action grounded in concrete page evidence?
- 1.0 = Action references exact bid/text/URL visible in the current observation
- 0.7 = Action targets a clearly identified element with minor ambiguity
- 0.4 = Action is reasonable but based on inference rather than direct observation
- 0.0 = Action is generic or speculative, no specific evidence from the page\
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

Example 1 (exploring — need more info):
{{
  "response": "Page shows Reviews section with 12 reviews total. Only 3 reviews are visible on screen. I found 1 matching reviewer so far (Catso). Progress: 3/12 reviews checked — incomplete. I need to scroll down to see remaining reviews before answering.",
  "action": "scroll(0, 500)",
  "additional_question": "In the screenshot, does a dark sub-menu appear with a Shipping link?",
  "relevance": 0.9,
  "confidence": 0.85,
  "surprise": 0.4
}}

Example 2 (final answer — retrieval, all data gathered):
{{
  "response": "I have scrolled through all 12 reviews. Found 4 reviewers mentioning the keyword: Alice, Bob, Carol, Dave. All content checked, no more reviews below. Ready to submit.",
  "action": "send_msg_to_user('{{\\"task_type\\": \\"RETRIEVE\\", \\"status\\": \\"SUCCESS\\", \\"retrieved_data\\": [\\"Alice\\", \\"Bob\\", \\"Carol\\", \\"Dave\\"]}}')",
  "additional_question": "",
  "relevance": 1.0,
  "confidence": 0.95,
  "surprise": 0.9
}}

Example 3 (final answer — navigation):
{{
  "response": "Current URL is /issues?state=open and the page displays the open issues list. This matches the objective. Navigation complete.",
  "action": "send_msg_to_user('{{\\"task_type\\": \\"NAVIGATE\\", \\"status\\": \\"SUCCESS\\", \\"retrieved_data\\": null}}')",
  "additional_question": "",
  "relevance": 1.0,
  "confidence": 0.9,
  "surprise": 0.8
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


_PROCESSOR_OUTPUT_KEYS = {"response", "action", "additional_question", "relevance", "confidence", "surprise"}


def _extract_json_safe(content: str) -> dict:
    """Attempt to extract a JSON object from raw LLM output."""
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.rfind("```")
        if start > 6 and end > start:
            try:
                obj = json.loads(content[start:end].strip())
                if isinstance(obj, dict) and obj.keys() & _PROCESSOR_OUTPUT_KEYS:
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.keys() & _PROCESSOR_OUTPUT_KEYS:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # JSON-style regex fallback: handles partially-formed JSON with quoted values
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
    if parsed:
        return parsed
    # Plain-text labeled-field fallback: handles "Reasoning: ...\nAction: ...\n" format
    plain: Dict = {}
    # Scores (unquoted numbers, e.g. "Relevance: 0.9")
    for key, label in [
        ("relevance", "Relevance"),
        ("confidence", "Confidence"),
        ("surprise", "Surprise"),
    ]:
        m = re.search(rf"^{label}:\s*([0-9.]+)", content, re.MULTILINE | re.IGNORECASE)
        if m:
            try:
                plain[key] = float(m.group(1))
            except ValueError:
                pass
    # Action: single line (may be a function call like scroll(0, 200))
    m = re.search(r"^Action:\s*(.+)$", content, re.MULTILINE)
    if m:
        plain["action"] = m.group(1).strip().strip('"')
    # Reasoning: everything from "Reasoning:" until the next labeled field
    m = re.search(
        r"^Reasoning:\s*(.*?)(?=\n(?:Action|Additional_question|Relevance|Confidence|Surprise):)",
        content,
        re.DOTALL | re.MULTILINE,
    )
    if m:
        plain["response"] = m.group(1).strip()
    # Additional_question: strip surrounding quotes if present
    m = re.search(r'^Additional_question:\s*"?(.*?)"?\s*$', content, re.MULTILINE)
    if m:
        plain["additional_question"] = m.group(1).strip()
    return plain


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

    if not action:
        m = re.search(
            r"((?:send_msg_to_user|click|fill|scroll|tab_focus|go_back|go_forward|goto|hover|press|select_option|new_tab|close_tab)\s*\(.*?\))",
            content,
            re.DOTALL,
        )
        if m:
            action = m.group(1).strip()

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
