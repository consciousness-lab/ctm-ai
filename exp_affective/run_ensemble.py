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


def run_instance(
    test_file, dataset, dataset_name,
    text_agent, audio_agent, video_agent,
    tracker, output_file,
):
    """Process one sample: load inputs -> 3 parallel agent calls -> majority vote -> save."""
    start_time = time.time()

    # Step 1: Load inputs
    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    system_prompt = inputs['system_prompt']
    label = inputs['label']
    muted_video_path = inputs['muted_video_path']
    audio_path = inputs['audio_path']

    query = f"{system_prompt}\n\ntarget text: '{target_sentence}'"

    print(f'--- Modality Ensemble for {test_file} ---')

    # Step 2: Run logic — 3 modality agents in parallel
    total_prompt_tokens = 0
    total_completion_tokens = 0

    def run_agent(agent_type):
        if agent_type == 'text':
            return agent_type, *text_agent.call(query)
        elif agent_type == 'audio':
            return agent_type, *audio_agent.call(query, audio_path=audio_path)
        elif agent_type == 'video':
            return agent_type, *video_agent.call(query, video_path=muted_video_path)

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
    print(f'  Result: {final_vote} | GT: {label_normalized} | {"✓" if is_correct else "✗"} ({duration:.1f}s)')

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
        '--dataset_name', type=str, default='urfunny',
        choices=['urfunny', 'mustard'], help='Dataset name (default: urfunny)',
    )
    parser.add_argument(
        '--provider', type=str, default='gemini',
        choices=['gemini', 'qwen'], help='LLM provider (default: gemini)',
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model name for litellm (default: auto based on provider)',
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Path to dataset JSON file (default: auto based on dataset_name)',
    )
    parser.add_argument(
        '--output', type=str, default=None, help='Output JSONL file path',
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature (default: 1.0)',
    )
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = args.output or f'ensemble_{args.dataset_name}_{args.provider}.jsonl'

    check_api_key(args.provider)
    litellm.set_verbose = False

    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )

    text_agent = create_agent(
        'text', provider=args.provider, model=args.model, temperature=args.temperature
    )
    audio_agent = create_agent(
        'audio', provider=args.provider, model=args.model, temperature=args.temperature
    )
    video_agent = create_agent(
        'video', provider=args.provider, model=args.model, temperature=args.temperature
    )
    print(f'Dataset: {args.dataset_name} | Provider: {args.provider} | Model: {text_agent.model}')
    print(f'Agents: TextAgent + AudioAgent + VideoAgent -> Majority Vote')

    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f'Resuming: {len(processed_keys)} done, {len(test_list) - len(processed_keys)} remaining')

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
            try:
                run_instance(
                    test_file, dataset, args.dataset_name,
                    text_agent, audio_agent, video_agent,
                    tracker, output_file,
                )
            except Exception as e:
                print(f'[ERROR] {test_file}: {e}')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary('Modality Ensemble (Text+Audio+Video)')
