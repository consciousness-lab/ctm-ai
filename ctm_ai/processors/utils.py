import json
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------


def _extract_json(content: str) -> dict:
    """Extract a JSON object from raw LLM output (handles ```json fences)."""
    if "```json" in content and "```" in content:
        start_idx = content.find("```json") + 7
        end_idx = content.rfind("```")
        if start_idx > 6 and end_idx > start_idx:
            return json.loads(content[start_idx:end_idx].strip())
    return json.loads(content)


def _extract_json_fallback(raw: str) -> dict:
    """
    Fallback: extract fields via regex when json.loads fails (e.g. unescaped
    quotes inside "response"). Fills in relevance, confidence, surprise,
    additional_questions, and response when possible.
    """
    parsed = {}
    # Numeric scores: "relevance": 1.0 or "relevance": 1
    for key in ("relevance", "confidence", "surprise"):
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
        if m:
            try:
                parsed[key] = float(m.group(1))
            except ValueError:
                pass
    # additional_questions: array of strings
    m = re.search(r'"additional_questions"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if m:
        questions_raw = m.group(1)
        questions = re.findall(r'"([^"]*)"', questions_raw)
        parsed["additional_questions"] = [q.strip() for q in questions if q.strip()]
    # Backward compatibility: additional_question (single)
    if "additional_questions" not in parsed:
        m = re.search(r'"additional_question"\s*:\s*"([^"]*)"', raw)
        if m:
            parsed["additional_questions"] = [m.group(1).strip()]
    # response: take everything between "response": " and ", "additional_question"
    # (handles unescaped quotes inside response by using next key as delimiter)
    resp_start_m = re.search(r'"response"\s*:\s*"', raw)
    if resp_start_m:
        start = resp_start_m.end()
        # Find end: ", "additional_questions" or ",\n    "additional_questions"
        end_m = re.search(r'",\s*"additional_questions"\s*:', raw[start:])
        if end_m:
            parsed["response"] = (
                raw[start : start + end_m.start()].replace('\\"', '"').strip()
            )
        else:
            parsed["response"] = raw[start:].replace('\\"', '"').rstrip().rstrip('"')
    return parsed


def parse_json_response(
    content: str, default_additional_question: str = ""
) -> tuple[str, str]:
    """
    Parse JSON response to extract response and additional_question.

    Args:
        content: The raw response content from LLM
        default_additional_question: Default additional question if parsing fails

    Returns:
        tuple: (parsed_content, additional_question)
    """
    try:
        parsed_response = _extract_json(content)
    except (json.JSONDecodeError, TypeError):
        raw = content
        if "```json" in content and "```" in content:
            start_idx = content.find("```json") + 7
            end_idx = content.rfind("```")
            if start_idx > 6 and end_idx > start_idx:
                raw = content[start_idx:end_idx].strip()
        parsed_response = _extract_json_fallback(raw)

    parsed_content = parsed_response.get("response", content)
    additional_question = parsed_response.get(
        "additional_question", default_additional_question
    )
    if additional_question == "":
        additional_question = default_additional_question

    return parsed_content, additional_question


def parse_json_response_with_scores(
    content: str,
    default_additional_questions: List[str] = None,
) -> Dict[str, object]:
    """Parse JSON response including self-evaluation scores.

    Args:
        content: Raw LLM output string.
        default_additional_questions: Fallback additional questions list.

    Returns:
        A dict containing:
        - 'response': the main answer
        - 'additional_questions': list of follow-up questions
        - 'relevance': float (0.0-1.0)
        - 'confidence': float (0.0-1.0)
        - 'surprise': float (0.0-1.0)
    """
    _DEFAULT_COMPONENT_SCORE = 0.5
    if default_additional_questions is None:
        default_additional_questions = []

    def _safe_float(val: object, lo: float, hi: float, default: float) -> float:
        try:
            return max(lo, min(hi, float(val)))
        except (ValueError, TypeError):
            return default

    try:
        parsed = _extract_json(content)
    except (json.JSONDecodeError, TypeError):
        # Fallback when JSON is invalid (e.g. unescaped quotes in "response")
        raw = content
        if "```json" in content and "```" in content:
            start_idx = content.find("```json") + 7
            end_idx = content.rfind("```")
            if start_idx > 6 and end_idx > start_idx:
                raw = content[start_idx:end_idx].strip()
        parsed = _extract_json_fallback(raw)
        if not parsed:
            parsed = {}
    response_text = parsed.get("response", content)

    # Handle additional_questions (list) or backward compatible additional_question (str)
    additional_questions = parsed.get("additional_questions")
    if additional_questions is None:
        # Backward compatibility
        old_q = parsed.get("additional_question", "")
        additional_questions = [old_q] if old_q else []
    if not additional_questions:
        additional_questions = default_additional_questions

    result: Dict[str, object] = {
        "response": response_text,
        "additional_questions": additional_questions,
    }

    for key in ("relevance", "confidence", "surprise"):
        result[key] = _safe_float(
            parsed.get(key, _DEFAULT_COMPONENT_SCORE),
            0.0,
            1.0,
            _DEFAULT_COMPONENT_SCORE,
        )

    return result
