import json
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Shared instruction fragments (DRY)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Initial Phase Preambles
# ---------------------------------------------------------------------------

_CONTEXT_PREAMBLE_WITH_CONTEXT = """## Your Task
Provide a comprehensive answer by combining:
1. **Your own modality observations** - What you can directly get from your inputs.
2. **Context information provided below** - Analyses from other processors that might be helpful, you should think further based on these context to answer the query.

IMPORTANT: The context below contains information from other modalities. You should actively use this context to enrich your answers.
Generate additional questions about information from other modalities you still need to answer the query(different from any previous questions)."""

_CONTEXT_PREAMBLE_NO_CONTEXT = """## Your Task
Answer the query using your modality-specific inputs (what you can directly observe from your video/audio/text/image inputs).

Generate additional questions to gather information from other modalities that could help answer the query more completely."""

_ADDITIONAL_QUESTIONS_INSTRUCTION = """
## Additional Questions Guidelines
Generate exactly 3 questions to gather information from OTHER modalities that would help answer the query:
Your additional_question should be potentially answerable by other modality about specific information that you are not sure about.
- Ask about specific, observable details (e.g., "What is the speaker's tone of voice?", "What facial expressions are shown?")
- Keep questions short and clean.
- Nothing else about the task or original query should be included."""

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


def build_json_format_score(has_context: bool = False) -> str:
    """Build JSON format instruction with appropriate context preamble.

    Args:
        has_context: Whether there is context history available (fuse_history or winner_answer)
    """
    preamble = (
        _CONTEXT_PREAMBLE_WITH_CONTEXT if has_context else _CONTEXT_PREAMBLE_NO_CONTEXT
    )

    return (
        preamble
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


# Keep backward compatibility - default to no context
JSON_FORMAT_SCORE = build_json_format_score(has_context=False)

# ---------------------------------------------------------------------------
# Link Form Phase - Evaluate if THIS processor can contribute NEW information
# Purpose: Determine if this processor should be linked to the winning processor
# ---------------------------------------------------------------------------
JSON_FORMAT_LINK_FORM = """
## Your Task
Answer the questions using your direct modality inputs and any context provided above.

## Output Format Rules
- Use numbered format: "1. [answer] 2. [answer] ..."
- SKIP questions you cannot answer (do NOT say "I cannot answer")
- Only include answers you CAN provide from your modality
- Keep the original question numbers even if you skip some
- If you cannot answer ANY question, response should be empty string ""

Example 1 - Can answer some questions:
{"response": "1. The tone is enthusiastic and upbeat. 3. The context appears to be a casual conversation.", "relevance": 0.7}

Example 2 - Cannot answer any question (all outside your modality and context):
{"response": "", "relevance": 0.0}

## Relevance Score (0.0 - 1.0)
Based on questions you COULD answer: How relevant do you think your response is to the question? 
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

Please respond in JSON format:
{"response": "Your numbered answers (or empty string if none)", "relevance": <number between 0.0 and 1.0>}"""

# ---------------------------------------------------------------------------
# Fuse Phase - Provide raw modality-specific information to other processors
# Purpose: Share your direct observations so other processors can use them as context
# ---------------------------------------------------------------------------
JSON_FORMAT_FUSE = """
## Your Task
Answer the questions using ONLY your direct modality inputs (what you can actually see/hear).

## Output Format Rules
- Use numbered format: "1. [answer] 2. [answer] ..."
- SKIP questions you cannot answer (do NOT include them at all)
- Only include answers you CAN provide from your modality
- Be specific and descriptive - your answers will help other processors
- If you cannot answer ANY question, response should be empty string ""

Example 1 - Can answer some questions:
{"response": "1. The tone is playful and amused with light chuckles. 2. The speaker seems to be responding to a fun challenge."}

Example 2 - Cannot answer any question:
{"response": ""}

Respond in JSON format: {"response": "Your numbered answers (or empty string if none)"}"""

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
