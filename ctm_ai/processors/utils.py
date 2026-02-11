import json
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Shared instruction fragments (DRY)
# ---------------------------------------------------------------------------

_CONTEXT_PREAMBLE = """You should utilize the provided information in the context and modality-specific information to answer the query.
There might have some answers to other queries, you should utilize them to answer the query. You should not generate the same additional questions as the previous ones."""

_ADDITIONAL_QUESTIONS_INSTRUCTION = """
Your additional_questions should be potentially answerable by other modality models about specific information that you are not sure about and the information you can not get from your observation/and context.
Each question should be just about what kind of information you need to get from other modality models, nothing else about the task or original query should be included. For example, what is the tone of the audio, what is the facial expression of the person, what is the caption of the image, etc. Each question needs to be short and clean.
Generate exactly 3 diverse questions targeting different aspects or modalities."""

_SCORE_RUBRIC = """
## Self-Evaluation Instructions

IMPORTANT: Evaluate ONLY the "response" field you wrote above. The "additional_question" must have NO influence on your scores.

First commit to your best answer in "response", then step back and critically self-assess along three dimensions:

### Relevance (0.0 - 1.0): How relevant do you think your response is to the question? 
Here, "relevant" means that the answer engages with the question and provides information 
that is useful or connected to addressing it. Even if the answer expresses uncertainty 
(e.g., "difficult to determine") but still explains reasoning, it should be considered relevant. 
Only answers that completely refuse, ignore, or go off-topic should be scored as 0.0. 
- 1.0 = Directly and precisely answers with specific details
- 0.8 = Mostly answers with useful supporting details
- 0.6 = Engages with the question but incomplete or limited
- 0.4 = Loosely connected, not very helpful
- 0.2 = Weak or indirect connection only
- 0.0 = Off-topic, refuses to answer, or irrelevant

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

Be honest and well-calibrated. Do NOT inflate scores. Most routine answers should score around relevance ~0.6, confidence ~0.6, surprise ~0.3."""

# ---------------------------------------------------------------------------
# JSON format template
# ---------------------------------------------------------------------------

JSON_FORMAT_SCORE = (
    _CONTEXT_PREAMBLE
    + """
Please respond in JSON format with the following structure:
{
    "response": "Your detailed response to the query",
    "additional_questions": ["question1", "question2", "question3"],
    "relevance": <number between 0.0 and 1.0>,
    "confidence": <number between 0.0 and 1.0>,
    "surprise": <number between 0.0 and 1.0>
}
"""
    + _ADDITIONAL_QUESTIONS_INSTRUCTION
    + _SCORE_RUBRIC
    + """

### Filling the score fields:
Assess your "response" (NOT "additional_questions") and fill in each field independently:
- "relevance": your relevance assessment (0.0 to 1.0)
- "confidence": your confidence assessment (0.0 to 1.0)
- "surprise": your surprise / novelty assessment (0.0 to 1.0)"""
)

# Simplified format for link_form phase - response + relevance
JSON_FORMAT_LINK_FORM = """
IMPORTANT: You can ONLY answer based on the modality-specific inputs you actually receive and the context information explicitly provided.

STRICT RULES:
1. If the question asks about a modality you do NOT have access to, you should not provide answer to that specific question" There will be multiple questions and you should only answer the questions that you have access to in the context and multimodal inputs, and give your relevance score based on these questions.
2. Do NOT infer, guess, or fabricate information about modalities you cannot observe.
3. Only use information directly visible/audible in your inputs or explicitly stated in the provided context.

First commit to your best answer in "response", then self-assess the relevance:

Relevance (0.0 - 1.0): How relevant do you think your response is to the question?
- 1.0 = Directly and precisely answers with specific details from your modality inputs
- 0.8 = Mostly answers with useful supporting details from your inputs
- 0.6 = Engages with the question but incomplete or limited
- 0.4 = Loosely connected, not very helpful
- 0.2 = Weak or indirect connection only
- 0.0 = You do NOT have access to the required modality, off-topic, or cannot answer

IMPORTANT: If you cannot answer because you lack the required modality input, relevance MUST be 0.0.
Be honest and well-calibrated. Do NOT inflate scores.

Please respond in JSON format:
{"response": "Your response to the query", "relevance": <number between 0.0 and 1.0>}"""

# Simplified format for fuse phase - only response needed
JSON_FORMAT_FUSE = """
IMPORTANT: You can ONLY answer based on the modality-specific inputs you actually receive. There will be multiple questions and you should only answer the questions that you have access to in the multimodal inputs.

STRICT RULES:
1. If the question asks about a modality you do NOT have access to, respond: "I cannot answer this question because I do not have access to [modality] information."
2. Do NOT infer, guess, or fabricate information about modalities you cannot observe.
3. Only use information directly visible/audible in your inputs.

Respond with the JSON format: {"response": "Your answer to the query"}"""

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
    additional_questions, and response when possible.
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
    # additional_questions: array of strings
    m = re.search(r'"additional_questions"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if m:
        questions_raw = m.group(1)
        questions = re.findall(r'"([^"]*)"', questions_raw)
        parsed['additional_questions'] = [q.strip() for q in questions if q.strip()]
    # Backward compatibility: additional_question (single)
    if 'additional_questions' not in parsed:
        m = re.search(r'"additional_question"\s*:\s*"([^"]*)"', raw)
        if m:
            parsed['additional_questions'] = [m.group(1).strip()]
    # response: take everything between "response": " and ", "additional_question"
    # (handles unescaped quotes inside response by using next key as delimiter)
    resp_start_m = re.search(r'"response"\s*:\s*"', raw)
    if resp_start_m:
        start = resp_start_m.end()
        # Find end: ", "additional_questions" or ",\n    "additional_questions"
        end_m = re.search(r'",\s*"additional_questions"\s*:', raw[start:])
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
        if '```json' in content and '```' in content:
            start_idx = content.find('```json') + 7
            end_idx = content.rfind('```')
            if start_idx > 6 and end_idx > start_idx:
                raw = content[start_idx:end_idx].strip()
        parsed = _extract_json_fallback(raw)
        if not parsed:
            parsed = {}
    response_text = parsed.get('response', content)

    # Handle additional_questions (list) or backward compatible additional_question (str)
    additional_questions = parsed.get('additional_questions')
    if additional_questions is None:
        # Backward compatibility
        old_q = parsed.get('additional_question', '')
        additional_questions = [old_q] if old_q else []
    if not additional_questions:
        additional_questions = default_additional_questions

    result: Dict[str, object] = {
        'response': response_text,
        'additional_questions': additional_questions,
    }

    for key in ('relevance', 'confidence', 'surprise'):
        result[key] = _safe_float(
            parsed.get(key, _DEFAULT_COMPONENT_SCORE),
            0.0,
            1.0,
            _DEFAULT_COMPONENT_SCORE,
        )

    return result
