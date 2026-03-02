"""
Unified ensemble: N votes from the same MultimodalAgent, then majority vote.

Each vote sends full video (with audio) + text. N votes run in parallel,
then majority vote determines the final answer.

Examples:
python run_unified_ensemble.py --dataset_name urfunny --provider qwen --n_votes 3
python run_unified_ensemble.py --dataset_name mustard --provider gemini --n_votes 3
"""

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    StatsTracker,
    check_api_key,
    create_agent,
    load_data,
    load_processed_keys,
    load_sample_inputs,
    normalize_label,
    save_result_to_jsonl,
)

sys.path.append('..')

N_VOTES = 3
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30


def extract_vote(answer):
    """Extract Yes/No from the answer."""
    if answer is None or not answer.strip():
        return 'Unknown'
    first_line = answer.strip().split('\n')[0].strip().lower()
    first_word = first_line.split()[0] if first_line.split() else ''
    if first_word == 'yes' or answer.strip().lower().startswith('yes'):
        return 'Yes'
    elif first_word == 'no' or answer.strip().lower().startswith('no'):
        return 'No'
    return 'Unknown'


def majority_vote(votes):
    """Return the majority vote result."""
    counts = Counter(votes)
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else 'Unknown'


def run_instance(test_file, dataset, dataset_name, agent, tracker, output_file):
    """Process one sample: load inputs -> N parallel votes -> majority vote -> save."""
    start_time = time.time()

    # Step 1: Load inputs
    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    system_prompt = inputs['system_prompt']
    label = inputs['label']
    full_video_path = inputs['full_video_path']

    query = f"{system_prompt}\n\ntarget text: '{target_sentence}'"

    print(f'--- Unified Ensemble ({N_VOTES} votes) for {test_file} ---')

    # Step 2: Run logic — N parallel multimodal votes
    total_prompt_tokens = 0
    total_completion_tokens = 0
    all_answers = []
    all_votes = []

    def single_vote(vote_idx):
        return vote_idx, *agent.call(query, video_path=full_video_path)

    with ThreadPoolExecutor(max_workers=N_VOTES) as executor:
        futures = {executor.submit(single_vote, i): i for i in range(N_VOTES)}
        for future in as_completed(futures):
            try:
                idx, answer, usage = future.result()
                all_answers.append(answer)
                vote = extract_vote(answer)
                all_votes.append(vote)
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                preview = answer.strip().replace('\n', ' ')[:80] if answer else 'None'
                print(f'  Vote {idx + 1}: {vote:<7} | {preview}...')
            except Exception as exc:
                print(f'  Vote: ERROR | {exc}')
                all_answers.append('Error')
                all_votes.append('Unknown')

    final_vote = majority_vote(all_votes)
    vote_counts = Counter(all_votes)

    end_time = time.time()
    duration = end_time - start_time
    tracker.add(duration, total_prompt_tokens, total_completion_tokens, N_VOTES)

    label_normalized = normalize_label(label)
    is_correct = final_vote == label_normalized
    print(
        f'  Result: {final_vote} | GT: {label_normalized} | {"✓" if is_correct else "✗"} ({duration:.1f}s)'
    )

    # Step 3: Save result
    result = {
        test_file: {
            'answer': [final_vote],
            'individual_answers': all_answers,
            'votes': all_votes,
            'final_vote': final_vote,
            'vote_distribution': dict(vote_counts),
            'label': label,
            'label_normalized': label_normalized,
            'correct': is_correct,
            'method': f'unified_ensemble_n{N_VOTES}',
            'usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'api_calls': N_VOTES,
            },
            'latency': duration,
        }
    }
    save_result_to_jsonl(result, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified Ensemble (N multimodal votes)'
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
        '--n_votes',
        type=int,
        default=3,
        help='Number of votes (default: 3)',
    )
    args = parser.parse_args()

    N_VOTES = args.n_votes

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = (
        args.output or f'unified_ensemble_{args.dataset_name}_{args.provider}.jsonl'
    )

    check_api_key(args.provider)
    litellm.set_verbose = False

    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )

    agent = create_agent(
        'multimodal',
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    print(
        f'Dataset: {args.dataset_name} | Provider: {args.provider} | Model: {agent.model}'
    )

    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(
            f'Resuming: {len(processed_keys)} done, {len(test_list) - len(processed_keys)} remaining'
        )

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
            try:
                run_instance(
                    test_file, dataset, args.dataset_name, agent, tracker, output_file
                )
            except Exception as e:
                print(f'[ERROR] {test_file}: {e}')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary(f'Unified Ensemble N={N_VOTES}')
