"""
Multimodal debate: 3 agents (Video, Audio, Text) debate over multiple rounds.

Each round, 3 agents analyze in parallel, then refine based on others' answers.
After all rounds, a judge (text-only) makes the final decision.

Examples:
python run_debate.py --dataset_name urfunny --provider qwen --rounds 3
python run_debate.py --dataset_name mustard --provider gemini --rounds 3
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    KeyRotator,
    StatsTracker,
    check_api_key,
    create_agent,
    load_data,
    load_processed_keys,
    load_sample_inputs,
    parse_api_keys,
    save_result_to_jsonl,
)

sys.path.append('..')

ROUNDS = 3
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30


def extract_answer(response):
    """Extract Yes/No from agent response.

    Tries (in order):
      1. Explicit 'my answer: yes/no' marker
      2. LAST line that starts with yes/no (CTM text prompt style:
         "context...\\n\\nYes, ...")
      3. Last non-empty line ending with yes/no
    """
    if response is None or not response.strip():
        return 'Unknown'
    lower = response.lower()
    # 1. Explicit marker
    if 'my answer: yes' in lower:
        return 'Yes'
    if 'my answer: no' in lower:
        return 'No'
    lines = [ln.strip() for ln in lower.splitlines() if ln.strip()]
    # 2. Scan from the END for a line starting with yes/no
    for line in reversed(lines):
        toks = line.split()
        if not toks:
            continue
        first_word = toks[0].rstrip('.!?,:;"\'')
        if first_word == 'yes':
            return 'Yes'
        if first_word == 'no':
            return 'No'
    # 3. Last line ends with yes/no
    if lines:
        last = lines[-1].rstrip('.!?"\' ')
        if last.endswith(' yes') or last == 'yes':
            return 'Yes'
        if last.endswith(' no') or last == 'no':
            return 'No'
    return 'Unknown'


def run_instance(
    test_file,
    dataset,
    dataset_name,
    video_agent,
    audio_agent,
    text_agent,
    tracker,
    output_file,
):
    """Process one sample: load inputs -> multi-round debate -> judge -> save."""
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0

    # Step 1: Load inputs
    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    label = inputs['label']
    muted_video_path = inputs['muted_video_path']
    audio_path = inputs['audio_path']
    config = inputs['config']

    prompts = config.get_debate_prompts()

    print(f'--- Debate for {test_file} ({ROUNDS} rounds) ---')

    # Step 2: Run logic — multi-round debate
    debate_history = ''
    video_answer = ''
    audio_answer = ''
    text_answer = ''

    def run_agent(agent_type, query):
        if agent_type == 'video':
            return agent_type, *video_agent.call(query, video_path=muted_video_path)
        elif agent_type == 'audio':
            return agent_type, *audio_agent.call(query, audio_path=audio_path)
        elif agent_type == 'text':
            return agent_type, *text_agent.call(query)

    for round_num in range(1, ROUNDS + 1):
        print(f'  Round {round_num}:')
        round_history = f'=== Round {round_num} ===\n\n'

        if round_num == 1:
            video_query = prompts['video_init']
            audio_query = prompts['audio_init']
            text_query = f"{prompts['text_init']}\n\ntarget text: '{target_sentence}'"
        else:
            video_query = prompts['video_refine'].format(
                own_answer=video_answer,
                audio_answer=audio_answer,
                text_answer=text_answer,
            )
            audio_query = prompts['audio_refine'].format(
                own_answer=audio_answer,
                video_answer=video_answer,
                text_answer=text_answer,
            )
            text_query = (
                f'{prompts["text_refine"].format(own_answer=text_answer, video_answer=video_answer, audio_answer=audio_answer)}'
                f"\n\ntarget text: '{target_sentence}'"
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_agent, 'video', video_query): 'video',
                executor.submit(run_agent, 'audio', audio_query): 'audio',
                executor.submit(run_agent, 'text', text_query): 'text',
            }
            results = {}
            for future in as_completed(futures):
                agent_type, response, usage = future.result()
                results[agent_type] = (response, usage)
                num_api_calls += 1
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)

        video_answer = results['video'][0] or ''
        audio_answer = results['audio'][0] or ''
        text_answer = results['text'][0] or ''

        round_history += f'[Video Expert]: {video_answer}\n\n'
        round_history += f'[Audio Expert]: {audio_answer}\n\n'
        round_history += f'[Text Expert]: {text_answer}\n\n'
        debate_history += round_history

        print(
            f'    Video: {extract_answer(video_answer)} | Audio: {extract_answer(audio_answer)} | Text: {extract_answer(text_answer)}'
        )

    # Judge makes final decision
    judge_query = f"{prompts['judge'].format(debate_history=debate_history)}\n\ntarget text: '{target_sentence}'"
    final_verdict, usage = text_agent.call(judge_query)
    num_api_calls += 1
    total_prompt_tokens += usage.get('prompt_tokens', 0)
    total_completion_tokens += usage.get('completion_tokens', 0)

    end_time = time.time()
    duration = end_time - start_time
    tracker.add(duration, total_prompt_tokens, total_completion_tokens, num_api_calls)

    print(
        f'  Judge: {final_verdict[:80] if final_verdict else "None"}... ({duration:.1f}s)'
    )

    # Step 3: Save result
    result = {
        test_file: {
            'answer': [final_verdict],
            'debate_history': debate_history,
            'final_votes': {
                'video': extract_answer(video_answer),
                'audio': extract_answer(audio_answer),
                'text': extract_answer(text_answer),
            },
            'label': label,
            'method': f'debate_{ROUNDS}rounds',
            'usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'api_calls': num_api_calls,
            },
            'latency': duration,
        }
    }
    save_result_to_jsonl(result, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3-Agent Multimodal Debate')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='urfunny',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: urfunny)',
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='gemini',
        choices=['gemini', 'qwen'],
        help='LLM provider (default: gemini)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name for litellm (default: auto based on provider)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset JSON file (default: auto based on dataset_name)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of debate rounds (default: 3)',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of samples to process concurrently (default: 4)',
    )
    parser.add_argument(
        '--api-keys',
        type=str,
        default=None,
        help='Comma-separated API keys to rotate across (default: env var)',
    )
    args = parser.parse_args()

    ROUNDS = args.rounds

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = args.output or f'debate_{args.dataset_name}_{args.provider}.jsonl'

    api_keys = parse_api_keys(args.api_keys, args.provider)
    check_api_key(args.provider, keys=api_keys)
    litellm.set_verbose = False

    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )
    rotator = KeyRotator(api_keys) if len(api_keys) > 1 else None

    video_agent = create_agent(
        'video',
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        api_key=api_keys[0] if api_keys else None,
        key_rotator=rotator,
    )
    audio_agent = create_agent(
        'audio',
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        api_key=api_keys[0] if api_keys else None,
        key_rotator=rotator,
    )
    text_agent = create_agent(
        'text',
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        api_key=api_keys[0] if api_keys else None,
        key_rotator=rotator,
    )
    print(
        f'Dataset: {args.dataset_name} | Provider: {args.provider} | Model: {text_agent.model} '
        f'| keys={len(api_keys)} | workers={args.max_workers}'
    )

    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(
            f'Resuming: {len(processed_keys)} done, {len(test_list) - len(processed_keys)} remaining'
        )

    pending = [tf for tf in test_list if tf not in processed_keys]
    progress_lock = threading.Lock()
    done_counter = [0]
    total_pending = len(pending)

    def worker(test_file):
        try:
            run_instance(
                test_file,
                dataset,
                args.dataset_name,
                video_agent,
                audio_agent,
                text_agent,
                tracker,
                output_file,
            )
        except Exception as e:
            print(f'[ERROR] {test_file}: {e}')
        finally:
            with progress_lock:
                done_counter[0] += 1
                if done_counter[0] % 10 == 0 or done_counter[0] == total_pending:
                    print(
                        f'[progress] {done_counter[0]}/{total_pending} samples done'
                    )

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(worker, tf) for tf in pending]
            for fut in as_completed(futures):
                fut.result()
    finally:
        tracker.print_summary(f'Debate - {ROUNDS} Rounds')
