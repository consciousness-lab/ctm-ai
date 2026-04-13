"""
AutoGen multi-agent sarcasm detection on MUSTARD.

Architecture: Tool-based agents in RoundRobinGroupChat
- 3 modality experts (video, audio, text) with analysis tools
- 1 judge agent for final decision
- Configurable debate rounds

Each expert calls their modality-specific tool (litellm multimodal LLM call)
on the first turn, then debates with other experts in subsequent turns.
The judge synthesizes all evidence and makes the final Yes/No decision.

Usage:
  python run_autogen.py --dataset_name mustard
  python run_autogen.py --dataset_name mustard --rounds 2 --max_workers 4
  python run_autogen.py --dataset_name mustard --instance_id 2_380  # single debug
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import litellm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dataset_configs import get_dataset_config
from llm_utils import (
    StatsTracker,
    check_api_key,
    create_agent as create_litellm_agent,
    load_data,
    load_processed_keys,
    load_sample_inputs,
    save_result_to_jsonl,
)

sys.path.append('..')

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL = 'gemini-2.5-flash-lite'
DEFAULT_ROUNDS = 2
GEMINI_OPENAI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'

# Cost tracking (gemini-2.5-flash-lite pricing per 1M tokens)
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30

# File write lock for thread-safe JSONL appends
_file_lock = threading.Lock()


# ============================================================================
# System Prompts
# ============================================================================

def _load_autogen_prompts(dataset_name='mustard'):
    """Load AutoGen prompts based on dataset."""
    config_map = {
        'mustard': 'sarcasm_ctm_config.json',
        'urfunny': 'urfunny_test_qwen_v12_config.json',
    }
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'ctm_conf',
        config_map.get(dataset_name, config_map['mustard']),
    )
    with open(config_path) as f:
        cfg = json.load(f)
    procs = cfg['processors_config']

    if dataset_name == 'urfunny':
        task = 'humor detection'
        yes_label = 'humorous'
        no_label = 'not humorous'
        query = 'Is the person being humorous or not?'
    else:
        task = 'sarcasm detection'
        yes_label = 'sarcastic'
        no_label = 'not sarcastic'
        query = 'Is the person being sarcastic or not?'

    video_expert = (
        f"{procs['video_processor']['system_prompt']}\n\n"
        f"In your FIRST turn, you MUST call the analyze_video tool to examine visual cues.\n"
        f"In subsequent turns, consider other experts' perspectives and refine your position.\n\n"
        f'Always end your response with "My Answer: Yes" ({yes_label}) or "My Answer: No" ({no_label}).'
    )
    audio_expert = (
        f"{procs['audio_processor']['system_prompt']}\n\n"
        f"In your FIRST turn, you MUST call the analyze_audio tool to examine vocal cues.\n"
        f"In subsequent turns, consider other experts' perspectives and refine your position.\n\n"
        f'Always end your response with "My Answer: Yes" ({yes_label}) or "My Answer: No" ({no_label}).'
    )
    text_expert = (
        f"{procs['language_processor']['system_prompt']}\n\n"
        f"In your FIRST turn, you MUST call the analyze_text tool to examine the dialogue.\n"
        f"In subsequent turns, consider other experts' perspectives and refine your position.\n\n"
        f'Always end your response with "My Answer: Yes" ({yes_label}) or "My Answer: No" ({no_label}).'
    )
    judge = (
        f'You are a {task} expert. You synthesize evidence from Video, Audio, and Text experts '
        f'to make a final determination.\n\n'
        f'Based solely on the analyses provided by the experts, determine if the person is being {yes_label}.\n\n'
        f'IMPORTANT: If the analyses express uncertainty, are inconclusive, or lack sufficient evidence, '
        f'you should answer "No" ({no_label}). Only answer "Yes" when there is clear, converging evidence.\n\n'
        f'Your response MUST contain exactly one of:\n'
        f'  FINAL ANSWER: Yes\n'
        f'  FINAL ANSWER: No'
    )
    video_tool_q = (
        f'{query} '
        'Analyze the visual cues, body language, facial expressions, and context. '
        'Note: This video has no audio — analyze visual cues only. '
        'Always ground your answer in specific visual observations. Be concise but thorough.'
    )
    audio_tool_q = (
        f'{query} '
        'Analyze the tone, emotion, pitch, speed, and vocal patterns. '
        'Pay special attention to the tone of voice, pitch variations, speaking speed, '
        'emphasis on certain words, and pauses and timing.'
    )
    text_tool_q = (
        f'{query} '
        'Analyze the text carefully. Be concise but thorough.\n\n'
        'Dialogue:\n{text}'
    )
    # Task instruction sent to the group chat (dataset-aware)
    task_instruction = (
        f'Determine if the following utterance is {yes_label} or not.\n\n'
        f'Dialogue:\n{{target_sentence}}\n\n'
        f'The last line is the target utterance. '
        f'Each expert should use their analysis tool to examine evidence, '
        f'then provide their assessment.'
    )
    # Tool docstrings (AutoGen uses these as tool descriptions sent to the LLM)
    video_tool_doc = (
        f'Analyze video frames for visual {yes_label} cues. Call this to examine '
        f"the speaker's facial expressions, body language, and gestures in the muted video."
    )
    audio_tool_doc = (
        f'Analyze audio for vocal {yes_label} cues. Call this to examine '
        f"the speaker's tone, pitch, pace, and emphasis."
    )
    text_tool_doc = (
        f'Analyze dialogue text for linguistic {yes_label} cues. Call this to examine '
        f'the utterance in its conversational context.'
    )
    return (
        video_expert, audio_expert, text_expert, judge,
        video_tool_q, audio_tool_q, text_tool_q,
        task_instruction, video_tool_doc, audio_tool_doc, text_tool_doc,
    )


# Defaults (overridden in main based on dataset_name)
(VIDEO_EXPERT_PROMPT, AUDIO_EXPERT_PROMPT, TEXT_EXPERT_PROMPT,
 JUDGE_PROMPT, VIDEO_TOOL_QUERY, AUDIO_TOOL_QUERY, TEXT_TOOL_QUERY,
 TASK_INSTRUCTION, VIDEO_TOOL_DOC, AUDIO_TOOL_DOC, TEXT_TOOL_DOC) = _load_autogen_prompts()




# ============================================================================
# Model Client
# ============================================================================


def _is_openai_model(model: str) -> bool:
    """Check if a model name is an OpenAI model."""
    openai_prefixes = ('o3', 'o4', 'gpt-', 'chatgpt-')
    return any(model.startswith(p) for p in openai_prefixes)


def create_model_client(model: str = DEFAULT_MODEL) -> OpenAIChatCompletionClient:
    """Create AutoGen model client. Auto-detects OpenAI vs Gemini from model name."""
    if _is_openai_model(model):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY not set')
        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': True,
                'family': 'unknown',
                'structured_output': True,
            },
        )
    else:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError('GEMINI_API_KEY not set')
        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=GEMINI_OPENAI_BASE_URL,
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': True,
                'family': 'unknown',
                'structured_output': False,
            },
        )


def _litellm_model_name(model: str) -> str:
    """Convert model name to litellm format."""
    if _is_openai_model(model):
        return model  # litellm recognizes OpenAI models directly
    return f'gemini/{model}'


# ============================================================================
# Core Logic
# ============================================================================


def extract_final_answer(messages) -> str:
    """Extract final Yes/No answer from debate messages."""
    # First: look for judge's FINAL ANSWER
    for msg in reversed(messages):
        content = getattr(msg, 'content', '')
        if not isinstance(content, str):
            continue
        if 'FINAL ANSWER' in content.upper():
            after = content.upper().split('FINAL ANSWER')[-1]
            if 'YES' in after[:20]:
                return 'Yes'
            elif 'NO' in after[:20]:
                return 'No'

    # Fallback: majority vote from last round of expert answers
    votes = {'Yes': 0, 'No': 0}
    for msg in reversed(messages):
        content = getattr(msg, 'content', '')
        source = getattr(msg, 'source', '')
        if not isinstance(content, str) or source == 'judge':
            continue
        lower = content.lower()
        if 'my answer: yes' in lower:
            votes['Yes'] += 1
        elif 'my answer: no' in lower:
            votes['No'] += 1
        if votes['Yes'] + votes['No'] >= 3:
            break

    if votes['Yes'] > votes['No']:
        return 'Yes'
    return 'No'


async def run_instance_async(
    test_file: str,
    dataset: dict,
    dataset_name: str,
    model: str,
    rounds: int,
    text_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Process one sample through AutoGen multi-agent debate.

    Args:
        text_model: If set, use a different model for text expert + text tool.
                    Other agents use `model`. Enables mixed-model ablation.
    """
    start_time = time.time()
    effective_text_model = text_model or model

    # API call tracking
    tool_api_calls = 0
    tool_prompt_tokens = 0
    tool_completion_tokens = 0

    # Load sample
    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    label = inputs['label']
    muted_video_path = inputs['muted_video_path']
    audio_path = inputs['audio_path']

    # ------------------------------------------------------------------
    # Define tools (closures capturing this sample's data)
    # ------------------------------------------------------------------

    def analyze_video() -> str:
        nonlocal tool_api_calls, tool_prompt_tokens, tool_completion_tokens
        agent = create_litellm_agent(
            'video', provider='gemini', model=_litellm_model_name(model)
        )
        result, usage = agent.call(VIDEO_TOOL_QUERY, video_path=muted_video_path)
        tool_api_calls += 1
        tool_prompt_tokens += usage.get('prompt_tokens', 0)
        tool_completion_tokens += usage.get('completion_tokens', 0)
        return result or 'No visual analysis available.'

    def analyze_audio() -> str:
        nonlocal tool_api_calls, tool_prompt_tokens, tool_completion_tokens
        agent = create_litellm_agent(
            'audio', provider='gemini', model=_litellm_model_name(model)
        )
        result, usage = agent.call(AUDIO_TOOL_QUERY, audio_path=audio_path)
        tool_api_calls += 1
        tool_prompt_tokens += usage.get('prompt_tokens', 0)
        tool_completion_tokens += usage.get('completion_tokens', 0)
        return result or 'No audio analysis available.'

    def analyze_text() -> str:
        nonlocal tool_api_calls, tool_prompt_tokens, tool_completion_tokens
        agent = create_litellm_agent(
            'text', provider='gemini', model=_litellm_model_name(effective_text_model)
        )
        query = TEXT_TOOL_QUERY.format(text=target_sentence)
        result, usage = agent.call(query)
        tool_api_calls += 1
        tool_prompt_tokens += usage.get('prompt_tokens', 0)
        tool_completion_tokens += usage.get('completion_tokens', 0)
        return result or 'No text analysis available.'

    # Dataset-aware tool descriptions (AutoGen passes these as function descriptions
    # to the LLM, so they must reflect the active task — sarcasm vs humor).
    analyze_video.__doc__ = VIDEO_TOOL_DOC
    analyze_audio.__doc__ = AUDIO_TOOL_DOC
    analyze_text.__doc__ = TEXT_TOOL_DOC

    # ------------------------------------------------------------------
    # Create AutoGen agents
    # ------------------------------------------------------------------

    base_model_client = create_model_client(model)
    text_model_client = (
        create_model_client(effective_text_model)
        if text_model
        else base_model_client
    )

    # reflect_on_tool_use=True makes the expert make a second LLM call after the
    # tool returns, so it can produce a real assessment ending with "My Answer: Yes/No".
    # Without this, AssistantAgent would return a raw ToolCallSummaryMessage and the
    # expert never gets to "speak" — its system prompt instructions are ignored.
    video_expert = AssistantAgent(
        name='video_expert',
        model_client=base_model_client,
        tools=[analyze_video],
        system_message=VIDEO_EXPERT_PROMPT,
        reflect_on_tool_use=True,
    )
    audio_expert = AssistantAgent(
        name='audio_expert',
        model_client=base_model_client,
        tools=[analyze_audio],
        system_message=AUDIO_EXPERT_PROMPT,
        reflect_on_tool_use=True,
    )
    text_expert = AssistantAgent(
        name='text_expert',
        model_client=text_model_client,
        tools=[analyze_text],
        system_message=TEXT_EXPERT_PROMPT,
        reflect_on_tool_use=True,
    )
    judge = AssistantAgent(
        name='judge',
        model_client=base_model_client,
        system_message=JUDGE_PROMPT,
    )

    # ------------------------------------------------------------------
    # Create team
    # ------------------------------------------------------------------

    # With reflect_on_tool_use=True each tool-calling turn produces:
    #   tool-request event + tool-result event + follow-up TextMessage
    # RoundRobinGroupChat counts all of these toward MaxMessageTermination, so
    # scale the cap generously: 8 messages/round covers 3 tool turns + 1 judge.
    termination = TextMentionTermination('FINAL ANSWER') | MaxMessageTermination(
        rounds * 8 + 4
    )

    team = RoundRobinGroupChat(
        [video_expert, audio_expert, text_expert, judge],
        termination_condition=termination,
    )

    # ------------------------------------------------------------------
    # Run debate
    # ------------------------------------------------------------------

    task = TASK_INSTRUCTION.format(target_sentence=target_sentence)

    result = await team.run(task=task, cancellation_token=CancellationToken())

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------

    # Count agent turns (non-user messages that have text content)
    agent_turns = 0
    for msg in result.messages:
        source = getattr(msg, 'source', '')
        if source and source != 'user':
            content = getattr(msg, 'content', '')
            if isinstance(content, str) and content.strip():
                agent_turns += 1

    # Build debate transcript
    debate_lines = []
    for msg in result.messages:
        source = getattr(msg, 'source', 'system')
        content = getattr(msg, 'content', '')
        if isinstance(content, str) and content.strip():
            debate_lines.append(f'[{source}]: {content}')
    debate_transcript = '\n\n'.join(debate_lines)

    # Extract final answer
    final_answer = extract_final_answer(result.messages)

    end_time = time.time()
    latency = end_time - start_time

    # Pull real AutoGen-side token usage from the model clients (cumulative over
    # this instance, since clients are created fresh per sample).
    def _client_usage(client):
        try:
            u = client.total_usage()
            return (getattr(u, 'prompt_tokens', 0) or 0,
                    getattr(u, 'completion_tokens', 0) or 0)
        except Exception:
            return (0, 0)

    base_pt, base_ct = _client_usage(base_model_client)
    if text_model and text_model_client is not base_model_client:
        text_pt, text_ct = _client_usage(text_model_client)
    else:
        text_pt, text_ct = (0, 0)
    autogen_prompt_tokens = base_pt + text_pt
    autogen_completion_tokens = base_ct + text_ct

    # autogen_model_calls: each agent_turn with a tool call is 2 LLM calls
    # (tool-decide + reflect), turns without tool use are 1.
    autogen_model_calls = agent_turns + tool_api_calls
    total_api_calls = tool_api_calls + autogen_model_calls
    total_prompt_tokens = tool_prompt_tokens + autogen_prompt_tokens
    total_completion_tokens = tool_completion_tokens + autogen_completion_tokens

    print(
        f'  {test_file}: {final_answer} (label={label}) | '
        f'api_calls={total_api_calls} '
        f'(tool={tool_api_calls}, autogen={autogen_model_calls}) | '
        f'tok={total_prompt_tokens + total_completion_tokens} | '
        f'{latency:.1f}s'
    )

    return {
        test_file: {
            'answer': [debate_transcript],
            'parsed_answer': [final_answer],
            'label': label,
            'method': f'autogen_tool_debate_{rounds}r'
            + (f'_text={effective_text_model}' if text_model else ''),
            'api_calls': total_api_calls,
            'tool_api_calls': tool_api_calls,
            'autogen_model_calls': autogen_model_calls,
            'agent_turns': agent_turns,
            'latency': latency,
            'usage': {
                'tool_prompt_tokens': tool_prompt_tokens,
                'tool_completion_tokens': tool_completion_tokens,
                'autogen_prompt_tokens': autogen_prompt_tokens,
                'autogen_completion_tokens': autogen_completion_tokens,
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
            },
        }
    }


def run_instance_sync(
    test_file: str,
    dataset: dict,
    dataset_name: str,
    model: str,
    rounds: int,
    text_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper — each thread gets its own event loop."""
    return asyncio.run(
        run_instance_async(
            test_file, dataset, dataset_name, model, rounds, text_model
        )
    )


# ============================================================================
# Thread-safe JSONL writer
# ============================================================================


def save_result_threadsafe(result: dict, output_file: str):
    """Thread-safe append to JSONL file."""
    with _file_lock:
        save_result_to_jsonl(result, output_file)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='AutoGen Multi-Agent Sarcasm Detection (Tool-based)'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='mustard',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: mustard)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model name (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset JSON file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=DEFAULT_ROUNDS,
        help=f'Debate rounds (default: {DEFAULT_ROUNDS})',
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='Max parallel workers (default: 4)',
    )
    parser.add_argument(
        '--text_model',
        type=str,
        default=None,
        help='Override model for text expert (e.g. o3). Others use --model.',
    )
    parser.add_argument(
        '--instance_id',
        type=str,
        default=None,
        help='Run single instance (debug mode)',
    )
    args = parser.parse_args()

    # Reload prompts for the target dataset
    import sys as _sys
    _this = _sys.modules[__name__]
    (_this.VIDEO_EXPERT_PROMPT, _this.AUDIO_EXPERT_PROMPT, _this.TEXT_EXPERT_PROMPT,
     _this.JUDGE_PROMPT, _this.VIDEO_TOOL_QUERY, _this.AUDIO_TOOL_QUERY, _this.TEXT_TOOL_QUERY,
     _this.TASK_INSTRUCTION, _this.VIDEO_TOOL_DOC, _this.AUDIO_TOOL_DOC,
     _this.TEXT_TOOL_DOC) = _load_autogen_prompts(args.dataset_name)

    config = get_dataset_config(args.dataset_name)
    dataset_path = args.dataset or config.get_default_dataset_path()
    tag = f'{args.rounds}r'
    if args.text_model:
        tag += f'_text-{args.text_model}'
    output_file = (
        args.output
        or f'autogen_{args.dataset_name}_{tag}.jsonl'
    )

    check_api_key('gemini')
    if args.text_model and _is_openai_model(args.text_model):
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY not set (needed for --text_model)')
    litellm.set_verbose = False

    dataset = load_data(dataset_path)
    model_desc = args.model
    if args.text_model:
        model_desc += f' (text: {args.text_model})'
    print(
        f'Dataset: {args.dataset_name} ({len(dataset)} samples) | '
        f'Model: {model_desc} | Rounds: {args.rounds}'
    )

    # Single instance debug mode
    if args.instance_id:
        if args.instance_id not in dataset:
            print(f'Instance {args.instance_id} not found in dataset')
            sys.exit(1)
        print(f'\n--- Debug: {args.instance_id} ---')
        result = run_instance_sync(
            args.instance_id, dataset, args.dataset_name,
            args.model, args.rounds, args.text_model,
        )
        data = result[args.instance_id]
        print(f'\nAnswer: {data["parsed_answer"][0]}')
        print(f'Label:  {data["label"]}')
        print(f'API Calls: {data["api_calls"]} (tool={data["tool_api_calls"]}, autogen={data["autogen_model_calls"]})')
        print(f'Latency: {data["latency"]:.1f}s')
        print(f'\n--- Debate Transcript ---\n{data["answer"][0][:2000]}')
        return

    # Full run
    test_list = list(dataset.keys())
    processed_keys = load_processed_keys(output_file)
    remaining = [k for k in test_list if k not in processed_keys]

    if processed_keys:
        print(
            f'Resuming: {len(processed_keys)} done, {len(remaining)} remaining'
        )

    if not remaining:
        print('All samples already processed.')
        return

    stats = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M,
        cost_output_per_1m=COST_OUTPUT_PER_1M,
    )

    start_total = time.time()
    completed = 0
    errors = 0

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for test_file in remaining:
                future = executor.submit(
                    run_instance_sync,
                    test_file,
                    dataset,
                    args.dataset_name,
                    args.model,
                    args.rounds,
                    args.text_model,
                )
                futures[future] = test_file

            for future in as_completed(futures):
                test_file = futures[future]
                try:
                    result = future.result()
                    save_result_threadsafe(result, output_file)
                    data = result[test_file]
                    stats.add(
                        data['latency'],
                        data['usage']['total_prompt_tokens'],
                        data['usage']['total_completion_tokens'],
                        data['api_calls'],
                    )
                    completed += 1
                except Exception as e:
                    errors += 1
                    print(f'[ERROR] {test_file}: {e}')
    finally:
        elapsed = time.time() - start_total
        print(f'\n{"=" * 60}')
        print(f'Completed: {completed} | Errors: {errors} | Total time: {elapsed:.1f}s')
        stats.print_summary(f'AutoGen Tool-Debate ({args.rounds} rounds)')
        print(f'Output: {output_file}')
        print(f'Evaluate: python calc_ctm_res.py {output_file} -d {args.dataset_name}')


if __name__ == '__main__':
    main()
