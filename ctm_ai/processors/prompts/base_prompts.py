"""
Base Prompts for CTM-AI Framework

This module contains base/common prompts used across different processor types,
including emotion recognition, sentiment analysis, search, math, and other tasks.
These prompts provide standardized JSON response formats and scoring rubrics.
"""

# ===========================================================================
# BASE PROMPTS - COMMON ACROSS PROCESSORS
# ===========================================================================

_BASE_CONTEXT_PREAMBLE = """You should utilize the information in the context history and modality-specific information to answer the query.
There might have some answers to other queries, you should utilize them to answer the query. You should not generate the same additional questions as the previous ones.
IMPORTANT: Your PRIMARY evidence should come from your own modality. Use other modalities' analyses as supplementary context, but do not let them override clear evidence from your own analysis. If your modality provides strong evidence, maintain your position even if other modalities disagree."""


def _build_base_additional_questions_instruction(num_questions: int = 3) -> str:
    if num_questions <= 0:
        return (
            '\nDo not generate any additional questions. '
            'Set additional_questions to an empty list [].'
        )
    return (
        '\nYour additional_questions should be potentially answerable by other modality '
        'models or other tools like search engine and about specific information that '
        'you are not sure about.\n'
        'Each question should be just about what kind of information you need to get '
        'from other modality models or other tools like search engine, nothing else '
        'about the task or original query should be included. For example, what is the '
        'tone of the audio, what is the facial expression of the person, what is the '
        'caption of the image, etc. Each question needs to be short and clean.\n'
        f'Generate exactly {num_questions} diverse questions targeting different '
        'aspects or modalities.'
    )


def _build_base_additional_questions_json(num_questions: int = 3) -> str:
    if num_questions <= 0:
        return '"additional_questions": []'
    questions = [f'"question{i}"' for i in range(1, num_questions + 1)]
    return f'"additional_questions": [{", ".join(questions)}]'


_BASE_SCORE_RUBRIC = """
## Self-Evaluation Instructions

Evaluate ONLY the "response" field you wrote above. The "additional_question" must have NO influence on your scores.

### STRICT CALIBRATION — READ BEFORE SCORING
These scores are used to RANK competing analyses against each other. If every analysis scores 0.9+, the ranking collapses and the system cannot pick the best one. **The full 0.0–1.0 range must be used.** Anchor yourself to:

- Most routine answers fall in **0.4–0.7** for relevance and for confidence.
- Scores of **0.9 or 1.0 are RESERVED** — they require specific, named evidence (see the per-dimension rules below).
- If you notice both relevance AND confidence trending ≥0.9, pause and lower one by at least 0.2 unless you can quote the specific textual / tonal / visual cues in your response.
- Ambiguity is the norm for sarcasm and humor — a confidence of 1.0 on a socially ambiguous case is almost always wrong.

### Relevance (0.0 - 1.0) — How directly does your response address the specific question?
- **1.0 (RARE)**: Commits to a clear verdict AND names ≥2 specific cues (specific words/phrases, specific tonal markers, specific visual features).
- **0.8**: Commits to a verdict and names ≥1 specific cue.
- **0.6**: Engages with the question and gives a reasoned opinion, but cues are described generally ("the tone seems off", "the expression looks odd") rather than named precisely.
- **0.4**: Mostly summarizes the context; verdict is weak, hedged, or implicit.
- **0.2**: Tangentially related; restates or describes without committing.
- **0.0**: Off-topic, refuses, or says "I cannot answer".

### Confidence (0.0 - 1.0) — How strong is your internal belief that the verdict is correct?
Sarcasm and humor are inherently ambiguous. A single clear signal should anchor to ~0.6, not ~1.0.
- **1.0 (VERY RARE)**: Multiple independent signals converge AND no contradicting evidence. Essentially "I would bet on this."
- **0.8**: Signals are clearly consistent; residual ambiguity is minor.
- **0.6**: One strong signal but non-trivial counter-signals exist; the call leans but is not certain.
- **0.4**: Truly mixed evidence; the call is a judgment rather than a deduction.
- **0.2**: Very uncertain; largely a guess.
- **0.0**: Cannot determine or the response says "I don't know".

### Surprise (0.0 - 1.0) — Does the verdict reverse the surface / literal reading?
- **1.0**: Full reversal — the literal reading would predict one thing, the correct verdict says the opposite (classic sarcasm override).
- **0.6**: Substantive twist — non-trivial reinterpretation required.
- **0.3**: Default / most obvious reading, no reversal.
- **0.0**: Literal restatement, no interpretive work done.

### Final calibration check (MANDATORY)
Before emitting the JSON, answer silently: "If I gave these scores to a hundred similar analyses, would they separate good ones from mediocre ones?" If your relevance is ≥0.9, you must be able to quote ≥2 specific cues in your response; if confidence is ≥0.9, list ≥2 mutually-reinforcing signals. Otherwise lower the score by at least 0.2. Typical well-calibrated scores for a solid-but-not-exceptional analysis: relevance ≈ 0.6, confidence ≈ 0.6, surprise ≈ 0.3."""


def build_base_score_format(num_questions: int = 3) -> str:
    """Build the JSON format prompt for initial phase with configurable question count."""
    questions_json = _build_base_additional_questions_json(num_questions)
    questions_instruction = _build_base_additional_questions_instruction(num_questions)
    return (
        _BASE_CONTEXT_PREAMBLE
        + f"""
Please respond in JSON format with the following structure:
{{
    "response": "Your detailed response to the query",
    {questions_json},
    "relevance": <number between 0.0 and 1.0>,
    "confidence": <number between 0.0 and 1.0>,
    "surprise": <number between 0.0 and 1.0>
}}
"""
        + questions_instruction
        + _BASE_SCORE_RUBRIC
        + """

### Filling the score fields:
Assess your "response" (NOT "additional_questions") and fill in each field independently:
- "relevance": your relevance assessment (0.0 to 1.0)
- "confidence": your confidence assessment (0.0 to 1.0)
- "surprise": your surprise / novelty assessment (0.0 to 1.0)"""
    )


BASE_JSON_FORMAT_SCORE = build_base_score_format(3)

BASE_JSON_FORMAT_LINK_FORM = """
You should utilize the information in the context history and modality-specific information to answer the query.
There might have some answers to other queries, you should utilize them to answer the query.
First commit to your best answer in "response", then step back and critically self-assess about the relevance of your answer to the question:

Relevance (0.0 - 1.0): How relevant do you think your response is to the question? 
Here, "relevant" means that the answer engages with the question and provides information 
that is useful or connected to addressing it. Even if the answer expresses uncertainty 
(e.g., "difficult to determine") but still explains reasoning, it should be considered relevant. 
Only answers that completely refuse, ignore, or go off-topic should be scored as 0.0. 
- 1.0 = Directly and precisely answers with specific details
- 0.8 = Mostly answers with useful supporting details

Be honest and well-calibrated. Do NOT inflate scores. Most routine answers should score around relevance ~0.6.

Please respond in JSON format:
{"response": "Your response to the query", "relevance": <number between 0.0 and 1.0>}"""

BASE_JSON_FORMAT_FUSE = """
Use the modality-specific information (audio tone, facial expressions, text sentiment, etc.) to answer the query.

Respond with the JSON format: {"response": "Your answer to the query"}"""
