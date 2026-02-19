DEFAULT_NUM_ADDITIONAL_QUESTIONS = 1

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

TOOLBENCH_SYSTEM_PROMPT = (
    "You are a tool agent designed to help users by utilizing available tools "
    "to answer their queries or complete tasks. You have access to various APIs "
    "and tools, and your job is to decide whether to call a tool or answer "
    "directly based on the context."
)

# ---------------------------------------------------------------------------
# Stage 1: Tool Decision Prompt (shared across all phases)
# ---------------------------------------------------------------------------

TOOLBENCH_TOOL_DECISION_PROMPT = """Task: {query}

You should utilize the tool `{function_name}` to help solve the task if it is relevant.
In the context below, there might be information from other tools or previous answers that might be helpful.

CONTEXT:
{context}

DECISION:
- If the tool helps even partially or it might be one of the steps to solve the task, CALL IT.
- If the tool does not help at all, or the context already provides enough information, answer directly.

OUTPUT PROTOCOL (MUST follow strictly):
- If you CALL the tool:
  - Return ONLY a function call via tool_calls.
  - Set assistant.content to null (no natural-language text).
  - Do NOT include any text explanation.
- If you DO NOT call the tool:
  - Return ONLY a natural-language answer in assistant.content.
  - Do NOT include tool_calls.
  - Include all context information above into a comprehensive answer.
"""

# ---------------------------------------------------------------------------
# Stage 2: Score Rubrics (private)
# ---------------------------------------------------------------------------

_SCORE_RUBRIC_FULL = """
## Self-Evaluation Instructions

IMPORTANT: Evaluate ONLY the "response" field you wrote above.

### Relevance (0.0 - 1.0): Does your response address the task with useful, specific information?
"Relevant" means the response engages with the task and provides concrete information that helps answer it. Even a partial answer with specific data (e.g. a number, a name, a fact from the tool) is highly relevant. Vague acknowledgments like "I called the tool successfully" without actual data are NOT relevant.
- 1.0 = Provides a direct, specific answer with concrete details (e.g. "The price is $142.50" or "The flight departs at 3pm")
- 0.8 = Answers the core question with useful details, may miss minor aspects
- 0.6 = Addresses the task but only partially — gives some useful data but leaves important parts unanswered
- 0.4 = Loosely related to the task; mostly generic text without specific data
- 0.2 = Barely connected; acknowledges the topic but provides no useful information
- 0.0 = Completely off-topic, refuses to answer, or empty

### Confidence (0.0 - 1.0): How certain are you about the information in your response?
- 1.0 = Very certain — response is based on clear, unambiguous tool output or well-established facts
- 0.8 = Mostly certain — tool output is clear but minor interpretation was needed
- 0.6 = Moderate certainty — some assumptions or inferences were made
- 0.4 = Significant uncertainty — tool output was ambiguous or incomplete
- 0.2 = Very uncertain — largely guessing with minimal supporting evidence
- 0.0 = Cannot determine — no reliable information available

### Surprise (0.0 - 1.0): Does your response bring novel information beyond what was already known in context?
- 1.0 = Reveals highly unexpected information or a non-obvious insight from combining tool results
- 0.6 = Mixes known information with some new findings from the tool
- 0.3 = Mostly confirms what was expected or already in context; incremental new data
- 0.0 = Adds nothing new — entirely repeats existing context or gives a standard/generic answer

Be honest and well-calibrated. Do NOT inflate scores."""

_SCORE_RUBRIC_RELEVANCE_ONLY = """
### Relevance (0.0 - 1.0): Does your response provide useful NEW information for the query?
"Relevant" here means the response provides specific, concrete information that helps answer the query AND goes beyond simply repeating what is already in the context. The purpose of this evaluation is to determine whether linking with this processor would bring additional value.
- 1.0 = Provides specific, concrete new information that directly answers the query (e.g. actual data, facts, or results not already known)
- 0.8 = Provides mostly useful new information with specific details; may overlap slightly with existing context
- 0.6 = Provides some new useful data but also repeats substantial parts of existing context
- 0.4 = Mostly repeats existing information with only minor new details
- 0.2 = Almost entirely repeats what is already known; negligible new contribution
- 0.0 = Completely off-topic, refuses to answer, or provides no information at all

Be honest and well-calibrated. Do NOT inflate scores."""

# ---------------------------------------------------------------------------
# Private Helper Functions
# ---------------------------------------------------------------------------


def _build_additional_questions_json_example(num_questions: int) -> str:
    """Build JSON example with the specified number of additional questions."""
    if num_questions <= 0:
        return '"additional_questions": []'
    questions = [f'"question{i}"' for i in range(1, num_questions + 1)]
    return f'"additional_questions": [{", ".join(questions)}]'


def _build_additional_questions_instruction(num_questions: int) -> str:
    """Build instruction for generating additional questions."""
    if num_questions <= 0:
        return (
            "Do not generate any additional questions. "
            "Set additional_questions to an empty list []."
        )
    return (
        "Your additional_questions should be potentially answerable by other tools "
        "like search engine and about specific information that you are not sure about.\n"
        "Your additional_questions should be just about what kind of information you "
        "need to get from other tools like search engine, nothing else about the task "
        "or original query should be included. For example, what is the weather in the "
        "city, what is the stock price of the company, etc. The question needs to be "
        "short and clean.\n"
        f"Generate exactly {num_questions} diverse question(s) targeting different "
        "aspects or tools."
    )


def _build_action_description(
    tool_called: bool,
    tool_name: str,
    tool_args: str,
    raw_result: str,
) -> str:
    """Describe what happened in Stage 1 so Stage 2 has full decision context."""
    if tool_called:
        return (
            f"You called tool `{tool_name}` with arguments: {tool_args}\n"
            f"The tool returned:\n{raw_result}"
        )
    return f"You analyzed the query and provided this direct assessment:\n{raw_result}"


def _build_context_section(
    fuse_history: list,
    winner_answer: list,
) -> str:
    """Build context section from fuse_history and winner_answer."""
    parts = []
    if fuse_history:
        parts.append("\nExtra information from other tools:")
        for i, item in enumerate(fuse_history, 1):
            parts.append(f'{i}. {item["processor_name"]}: {item["answer"]}')
    if winner_answer:
        parts.append("\nPrevious answers to the same query:")
        for i, item in enumerate(winner_answer, 1):
            parts.append(f'{i}. {item["processor_name"]}: {item["answer"]}')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 2: Unified Synthesis Prompt Builder (public API)
# ---------------------------------------------------------------------------


def build_tool_stage2_prompt(
    query: str,
    tool_called: bool,
    tool_name: str,
    tool_args: str,
    raw_result: str,
    phase: str,
    fuse_history: list = None,
    winner_answer: list = None,
    num_additional_questions: int = DEFAULT_NUM_ADDITIONAL_QUESTIONS,
) -> str:
    """
    Build the Stage 2 synthesis prompt for all phases.

    After Stage 1 (tool decision + execution), this prompt asks the LLM to:
      - initial:   synthesize answer + additional_questions + full scores
      - link_form: synthesize answer + relevance score
      - fuse:      synthesize answer only
    """
    action_desc = _build_action_description(
        tool_called, tool_name, tool_args, raw_result
    )

    if phase == "initial":
        context = _build_context_section(fuse_history or [], winner_answer or [])
        json_example = _build_additional_questions_json_example(
            num_additional_questions
        )
        questions_instruction = _build_additional_questions_instruction(
            num_additional_questions
        )
        context_block = f"\n{context}\n" if context else ""

        return f"""Regarding the task: {query}

{action_desc}
{context_block}
IMPORTANT: Your response MUST be a comprehensive, self-contained answer to the task.
- You MUST combine the tool result above with ALL context information (extra information from other tools AND previous answers) to produce a COMPLETE answer.
- Do NOT only describe what you newly observed from the tool call. Instead, integrate ALL available information into one coherent, thorough response.
- Provide specific details and data whenever possible. Do not just say you successfully called a tool.
- If you think you need information from other APIs or tools, generate additional questions about what results you need.

Please respond in JSON format:
{{
    "response": "Your detailed response to the task",
    {json_example},
    "relevance": <number between 0.0 and 1.0>,
    "confidence": <number between 0.0 and 1.0>,
    "surprise": <number between 0.0 and 1.0>
}}

{questions_instruction}
{_SCORE_RUBRIC_FULL}"""

    if phase == "link_form":
        context = _build_context_section(fuse_history or [], winner_answer or [])
        context_block = f"\n{context}\n" if context else ""

        return f"""Regarding the query: {query}

{action_desc}
{context_block}
IMPORTANT: Your response MUST combine ALL available information — the tool result above, extra information from other tools, and any previous answers — into a COMPLETE answer to the query.
Do NOT only report what you newly obtained. Integrate everything you know into one coherent response, then evaluate its relevance.

Please respond in JSON format:
{{
    "response": "Your comprehensive response integrating all available information",
    "relevance": <number between 0.0 and 1.0>
}}
{_SCORE_RUBRIC_RELEVANCE_ONLY}"""

    # fuse phase (or any other)
    return f"""Regarding the query: {query}

{action_desc}

Based on the above, generate a response that answers the query.

Please respond in JSON format:
{{"response": "Your response based on the available information"}}"""
