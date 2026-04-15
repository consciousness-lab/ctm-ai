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
