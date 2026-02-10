import json
import re
from typing import Dict

# ---------------------------------------------------------------------------
# Shared instruction fragments (DRY)
# ---------------------------------------------------------------------------

_CONTEXT_PREAMBLE = """You should utilize the information in the context history and modality-specific information to answer the query.
There might have some answers to other queries, you should utilize them to answer the query. You should not generate the same additional questions as the previous ones."""

_ADDITIONAL_QUESTION_INSTRUCTION = """
Your additional_question should be potentially answerable by other modality models or other tools like search engine and about specific information that you are not sure about.
Your additional_question should be just about what kind of information you need to get from other modality models or other tools like search engine, nothing else about the task or original query should be included. For example, what is the tone of the audio, what is the facial expression of the person, what is the caption of the image, etc. The question needs to be short and clean."""

_SCORE_RUBRIC = """
## Self-Evaluation Instructions

IMPORTANT: Evaluate ONLY the "response" field you wrote above. The "additional_question" must have NO influence on your scores.

First commit to your best answer in "response", then step back and critically self-assess along three dimensions:

### Relevance (0.0 - 1.0): How well does your response address the query?
- 1.0 = Directly and precisely answers with specific details
- 0.8 = Mostly answers with useful supporting details
- 0.6 = Engages with the question but incomplete or limited
- 0.4 = Loosely connected, not very helpful
- 0.2 = Weak or indirect connection only
- 0.0 = Off-topic, refuses to answer, or irrelevant
(Note: Expressing uncertainty while still providing reasoning counts as relevant, ~0.6+)

### Confidence (0.0 - 1.0): How certain are you about your response?
- 1.0 = Very certain, definitive statements
- 0.8 = Mostly certain, minor qualifications
- 0.6 = Some uncertainty but reasonable conclusions
- 0.4 = Significant uncertainty, many qualifications
- 0.2 = Very uncertain, extensive hedging
- 0.0 = Cannot determine, "I don't know", or refuses
(Note: If your response says "cannot determine" or equivalent, this MUST be 0.0)

### Surprise (0.0 - 1.0): How novel or insightful is your response?
- 1.0 = Highly unexpected, novel insights
- 0.6 = Mix of predictable and unexpected
- 0.3 = Mostly predictable, common knowledge
- 0.0 = Entirely expected, standard response

Be honest and well-calibrated. Do NOT inflate scores. Most routine answers should score around relevance ~0.7, confidence ~0.7, surprise ~0.3."""

# ---------------------------------------------------------------------------
# JSON format template
# ---------------------------------------------------------------------------

JSON_FORMAT_SCORE = (
    _CONTEXT_PREAMBLE
    + """
Please respond in JSON format with the following structure:
{
    "response": "Your detailed response to the query",
    "additional_question": "A question for other modality models or tools if you need more information.",
    "relevance": <number between 0.0 and 1.0>,
    "confidence": <number between 0.0 and 1.0>,
    "surprise": <number between 0.0 and 1.0>
}
"""
    + _ADDITIONAL_QUESTION_INSTRUCTION
    + _SCORE_RUBRIC
    + """

### Filling the score fields:
Assess your "response" (NOT "additional_question") and fill in each field independently:
- "relevance": your relevance assessment (0.0 to 1.0)
- "confidence": your confidence assessment (0.0 to 1.0)
- "surprise": your surprise / novelty assessment (0.0 to 1.0)"""
)

# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------


def _extract_json(content: str) -> dict:
    """Extract a JSON object from raw LLM output (handles ```json fences)."""
    if '```json' in content and '```' in content:
        start_idx = content.find('```json') + 7
        end_idx = content.rfind('```')
        if start_idx > 6 and end_idx > start_idx:
            return json.loads(content[start_idx:end_idx].strip())
    return json.loads(content)


def _extract_json_fallback(raw: str) -> dict:
    """
    Fallback: extract fields via regex when json.loads fails (e.g. unescaped
    quotes inside "response"). Fills in relevance, confidence, surprise,
    additional_question, and response when possible.
    """
    parsed = {}
    # Numeric scores: "relevance": 1.0 or "relevance": 1
    for key in ('relevance', 'confidence', 'surprise'):
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
        if m:
            try:
                parsed[key] = float(m.group(1))
            except ValueError:
                pass
    # additional_question: short string, often no internal quotes
    m = re.search(r'"additional_question"\s*:\s*"([^"]*)"', raw)
    if m:
        parsed['additional_question'] = m.group(1).strip()
    # response: take everything between "response": " and ", "additional_question"
    # (handles unescaped quotes inside response by using next key as delimiter)
    resp_start_m = re.search(r'"response"\s*:\s*"', raw)
    if resp_start_m:
        start = resp_start_m.end()
        # Find end: ", "additional_question" or ",\n    "additional_question"
        end_m = re.search(r'",\s*"additional_question"\s*:', raw[start:])
        if end_m:
            parsed['response'] = (
                raw[start : start + end_m.start()].replace('\\"', '"').strip()
            )
        else:
            parsed['response'] = raw[start:].replace('\\"', '"').rstrip().rstrip('"')
    return parsed


def parse_json_response(
    content: str, default_additional_question: str = ''
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
        if '```json' in content and '```' in content:
            start_idx = content.find('```json') + 7
            end_idx = content.rfind('```')
            if start_idx > 6 and end_idx > start_idx:
                raw = content[start_idx:end_idx].strip()
        parsed_response = _extract_json_fallback(raw)

    parsed_content = parsed_response.get('response', content)
    additional_question = parsed_response.get(
        'additional_question', default_additional_question
    )
    if additional_question == '':
        additional_question = default_additional_question

    return parsed_content, additional_question


def parse_json_response_with_scores(
    content: str,
    default_additional_question: str = '',
) -> Dict[str, object]:
    """Parse JSON response including self-evaluation scores.

    Args:
        content: Raw LLM output string.
        default_additional_question: Fallback additional question.

    Returns:
        A dict containing:
        - 'response': the main answer
        - 'additional_question': follow-up question
        - 'relevance': float (0.0-1.0)
        - 'confidence': float (0.0-1.0)
        - 'surprise': float (0.0-1.0)
    """
    _DEFAULT_COMPONENT_SCORE = 0.5

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
        if '```json' in content and '```' in content:
            start_idx = content.find('```json') + 7
            end_idx = content.rfind('```')
            if start_idx > 6 and end_idx > start_idx:
                raw = content[start_idx:end_idx].strip()
        parsed = _extract_json_fallback(raw)
        if not parsed:
            parsed = {}
    response_text = parsed.get('response', content)

    additional_question = parsed.get('additional_question', default_additional_question)
    if not additional_question:
        additional_question = default_additional_question

    result: Dict[str, object] = {
        'response': response_text,
        'additional_question': additional_question,
    }

    for key in ('relevance', 'confidence', 'surprise'):
        result[key] = _safe_float(
            parsed.get(key, _DEFAULT_COMPONENT_SCORE),
            0.0,
            1.0,
            _DEFAULT_COMPONENT_SCORE,
        )

    return result
