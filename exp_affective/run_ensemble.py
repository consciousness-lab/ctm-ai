"""
Modality ensemble: TextAgent + AudioAgent + VideoAgent, then majority vote.

Each agent independently analyzes its own modality, then majority vote
across the 3 results determines the final answer.

Examples:
python run_ensemble.py --dataset_name urfunny --provider qwen
python run_ensemble.py --dataset_name mustard --provider gemini
"""

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading

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
    normalize_label,
    parse_api_keys,
    save_result_to_jsonl,
)

sys.path.append('..')

COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30


def extract_vote(answer):
    """Extract Yes/No from the answer.

    The CTM-aligned text system prompt asks the agent to "first identify the
    CONTEXT, then analyze", so responses often have two paragraphs:
      1. "The context is ..."  (not a yes/no answer)
      2. "Yes, <reasoning>..."  OR  "<Speaker>'s statement is sarcastic..."

    Tries (in order):
      1. Explicit 'my answer: yes/no' marker
      2. LAST line that starts with yes/no
      3. Last non-empty line ends with yes/no
      4. Phrase-based fallback for task-specific conclusions
         ('not sarcastic' / 'is sarcastic' / 'is humorous' / 'not humorous')
    """
    if answer is None or not answer.strip():
        return 'Unknown'
    lower = answer.strip().lower()
    # 1. Explicit marker (highest priority)
    if 'my answer: yes' in lower:
        return 'Yes'
    if 'my answer: no' in lower:
        return 'No'
    lines = [ln.strip() for ln in lower.splitlines() if ln.strip()]
    # 2. Scan from the END for the first line that starts with yes/no
    for line in reversed(lines):
        toks = line.split()
        if not toks:
            continue
        first_word = toks[0].rstrip('.!?,:;"\'')
        if first_word == 'yes':
            return 'Yes'
        if first_word == 'no':
            return 'No'
    # 3. Last line ends with yes/no (rare: "the answer is yes")
    if lines:
        last = lines[-1].rstrip('.!?"\' ')
        if last.endswith(' yes') or last == 'yes':
            return 'Yes'
        if last.endswith(' no') or last == 'no':
            return 'No'
    # 4. Phrase-based fallback (task-specific). Check negations first so
    #    "not sarcastic" doesn't trigger "sarcastic".
    negations = (
        'not sarcastic', "isn't sarcastic", 'is not sarcastic',
        'not being sarcastic', 'no sarcasm', 'not humorous',
        "isn't humorous", 'is not humorous', 'not being humorous',
        'not a joke', 'not humor',
    )
    affirmatives = (
        'is sarcastic', 'being sarcastic', 'are sarcastic',
        'sarcasm is present', 'a form of sarcasm', 'is sarcasm',
        'is humorous', 'being humorous', 'are humorous',
        'is humor', 'a joke', 'humor is present',
    )
    if any(n in lower for n in negations):
        return 'No'
    if any(a in lower for a in affirmatives):
        return 'Yes'
    return 'Unknown'


def majority_vote(votes):
    """Return the majority vote, ignoring Unknown votes.

    If all votes are Unknown, returns Unknown. Otherwise picks the most
    common valid vote; ties are broken in insertion order (so a consistent
    agent ordering produces consistent tie-breaks across runs).
    """
    valid = [v for v in votes if v != 'Unknown']
    if not valid:
        return 'Unknown'
    return Counter(valid).most_common(1)[0][0]




def run_instance(
    test_file,
    dataset,
    dataset_name,
    text_agent,
    audio_agent,
    video_agent,
    tracker,
    output_file,
):
    """Process one sample: load inputs -> 3 parallel agent calls -> majority vote -> save."""
    start_time = time.time()

    # Step 1: Load inputs
    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    label = inputs['label']
    muted_video_path = inputs['muted_video_path']
    audio_path = inputs['audio_path']
    config = inputs['config']

    # Use CTM-aligned modality-specific system prompts + task query, so ensemble
    # starts from the same expert knowledge as debate/orchestra/CTM.
    mod = config.get_modality_system_prompts()
    task_query = (
        f'Task: {config.get_task_query()}\n'
        "Your answer must start with 'Yes' or 'No', followed by your reasoning. "
        'If you are unsure or the evidence is inconclusive, answer "No".'
    )
    text_query = f"{mod['text']}\n\n{task_query}\n\ntarget text: '{target_sentence}'"
    audio_query = f"{mod['audio']}\n\n{task_query}"
    video_query = f"{mod['video']}\n\n{task_query}"

    print(f'--- Modality Ensemble for {test_file} ---')

    # Step 2: Run logic — 3 modality agents in parallel
    total_prompt_tokens = 0
    total_completion_tokens = 0

    def run_agent(agent_type):
        if agent_type == 'text':
            return agent_type, *text_agent.call(text_query)
        elif agent_type == 'audio':
            return agent_type, *audio_agent.call(audio_query, audio_path=audio_path)
        elif agent_type == 'video':
            return agent_type, *video_agent.call(video_query, video_path=muted_video_path)

    agent_types = ['text', 'audio', 'video']
    all_answers = {}
    all_votes = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_agent, t): t for t in agent_types}
        for future in as_completed(futures):
            atype = futures[future]
            try:
                agent_type, answer, usage = future.result()
                all_answers[agent_type] = answer
                vote = extract_vote(answer)
                all_votes[agent_type] = vote
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                preview = answer.strip().replace('\n', ' ')[:80] if answer else 'None'
                print(f'  [{agent_type:>5}]: {vote:<7} | {preview}...')
            except Exception as exc:
                print(f'  [{atype:>5}]: ERROR | {exc}')
                all_answers[atype] = 'Error'
                all_votes[atype] = 'Unknown'

    votes_list = [all_votes.get(t, 'Unknown') for t in agent_types]
    final_vote = majority_vote(votes_list)
    vote_counts = Counter(votes_list)

    end_time = time.time()
    duration = end_time - start_time
    tracker.add(duration, total_prompt_tokens, total_completion_tokens, 3)

    label_normalized = normalize_label(label)
    is_correct = final_vote == label_normalized
    print(
        f'  Result: {final_vote} | GT: {label_normalized} | {"✓" if is_correct else "✗"} ({duration:.1f}s)'
    )

    # Step 3: Save result
    result = {
        test_file: {
            'answer': [final_vote],
            'individual_answers': {t: all_answers.get(t) for t in agent_types},
            'votes': {t: all_votes.get(t) for t in agent_types},
            'final_vote': final_vote,
            'vote_distribution': dict(vote_counts),
            'label': label,
            'label_normalized': label_normalized,
            'correct': is_correct,
            'method': 'modality_ensemble',
            'usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'api_calls': 3,
            },
            'latency': duration,
        }
    }
    save_result_to_jsonl(result, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modality Ensemble (Text+Audio+Video majority voting)'
    )
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

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = args.output or f'ensemble_{args.dataset_name}_{args.provider}.jsonl'

    api_keys = parse_api_keys(args.api_keys, args.provider)
    check_api_key(args.provider, keys=api_keys)
    litellm.set_verbose = False

    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )
    rotator = KeyRotator(api_keys) if len(api_keys) > 1 else None

    text_agent = create_agent(
        'text',
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
    video_agent = create_agent(
        'video',
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        api_key=api_keys[0] if api_keys else None,
        key_rotator=rotator,
    )
    print(
        f'Dataset: {args.dataset_name} | Provider: {args.provider} | Model: {text_agent.model}'
    )
    print(
        f'Agents: TextAgent + AudioAgent + VideoAgent -> Majority Vote '
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
                text_agent,
                audio_agent,
                video_agent,
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
        tracker.print_summary('Modality Ensemble (Text+Audio+Video)')
