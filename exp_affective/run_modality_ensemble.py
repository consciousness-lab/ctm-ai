"""
Modality Ensemble experiment - majority voting across 3 modality-specific agents.

Creates TextAgent, AudioAgent, and VideoAgent. Each independently analyzes the sample,
then majority vote across the 3 results determines the final answer.

Examples:
python run_modality_ensemble.py --dataset_name urfunny --provider gemini --output modality_ensemble_urfunny_gemini.jsonl
python run_modality_ensemble.py --dataset_name mustard --provider qwen --output modality_ensemble_mustard_qwen.jsonl
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
    get_audio_path,
    get_muted_video_path,
    load_data,
    load_processed_keys,
    normalize_label,
    save_result_to_jsonl,
)

sys.path.append('..')

# Pricing for Gemini 2.0 Flash Lite
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30


def extract_vote(answer):
    """
    Extract Yes/No from the answer with strict format control.

    Expects the answer to start with 'Yes' or 'No' (case-insensitive).
    """
    if answer is None or not answer.strip():
        return 'Unknown'

    answer_stripped = answer.strip()
    answer_lower = answer_stripped.lower()

    # Strategy 1: Check if the first line starts with Yes/No
    first_line = answer_stripped.split('\n')[0].strip().lower()
    first_word = first_line.split()[0] if first_line.split() else ''

    if first_word == 'yes':
        return 'Yes'
    elif first_word == 'no':
        return 'No'

    # Strategy 2: Check if answer starts with Yes/No
    if answer_lower.startswith('yes'):
        return 'Yes'
    elif answer_lower.startswith('no'):
        return 'No'

    # Strategy 3: Fallback - check first 20 characters
    first_chars = answer_lower[:20]
    if first_chars.startswith('yes'):
        return 'Yes'
    elif first_chars.startswith('no'):
        return 'No'

    print(
        f'  [WARNING] Could not extract clear Yes/No from answer: {answer_stripped[:100]}...'
    )
    return 'Unknown'


def majority_vote(votes):
    """Return the majority vote result"""
    vote_counts = Counter(votes)
    most_common = vote_counts.most_common(1)
    if most_common:
        return most_common[0][0]
    return 'Unknown'


def run_instance(
    test_file,
    dataset,
    dataset_name,
    text_agent,
    audio_agent,
    video_agent,
    tracker,
    output_file='modality_ensemble.jsonl',
):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0

    config = get_dataset_config(dataset_name)
    sample = dataset[test_file]

    target_sentence = config.get_text_field(sample)
    system_prompt = config.get_system_prompt()

    # Prepare query with target sentence
    query = f"{system_prompt}\n\ntarget text: '{target_sentence}'"

    # Paths for audio and video
    audio_path = get_audio_path(test_file, dataset_name)
    video_path = get_muted_video_path(test_file, dataset_name)

    print(f'--- Modality Ensemble for {test_file} ---')

    def run_agent(agent_type):
        """Run a single modality agent and return results"""
        if agent_type == 'text':
            answer, usage = text_agent.call(query)
        elif agent_type == 'audio':
            answer, usage = audio_agent.call(query, audio_path=audio_path)
        elif agent_type == 'video':
            answer, usage = video_agent.call(query, video_path=video_path)
        else:
            raise ValueError(f'Unknown agent type: {agent_type}')
        return agent_type, answer, usage

    agent_types = ['text', 'audio', 'video']
    all_answers = {}
    all_votes = {}

    # Run all 3 agents in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_agent, t): t for t in agent_types}

        for future in as_completed(futures):
            agent_type = futures[future]
            try:
                atype, answer, usage = future.result()
                all_answers[atype] = answer
                vote = extract_vote(answer)
                all_votes[atype] = vote

                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                num_api_calls += 1

                answer_preview = (
                    answer.strip().replace('\n', ' ')[:100] if answer else 'None'
                )
                print(f'  [{atype:>5}]: {vote:<7} | {answer_preview}...')

            except Exception as exc:
                print(f'  [{agent_type:>5}]: ERROR   | Exception: {exc}')
                all_answers[agent_type] = 'Error'
                all_votes[agent_type] = 'Unknown'

    # Majority voting across 3 modality agents
    votes_list = [all_votes.get(t, 'Unknown') for t in agent_types]
    final_vote = majority_vote(votes_list)
    vote_counts = Counter(votes_list)
    vote_distribution = ', '.join([f'{v}: {c}' for v, c in vote_counts.most_common()])
    final_verdict = f'{final_vote} (Distribution: {vote_distribution})'

    end_time = time.time()
    duration = end_time - start_time

    tracker.add(duration, total_prompt_tokens, total_completion_tokens, num_api_calls)

    # Normalize label for comparison
    ground_truth = config.get_label_field(sample)
    ground_truth_normalized = normalize_label(ground_truth)
    is_correct = final_vote == ground_truth_normalized
    match_symbol = '✓' if is_correct else '✗'

    print('------------------------------------------')
    print(f'Final Verdict: {final_verdict}')
    print(f'Ground Truth:  {ground_truth} (normalized: {ground_truth_normalized})')
    print(f'Match:         {match_symbol} ({final_vote} vs {ground_truth_normalized})')
    print(
        f'Time: {duration:.2f}s | API Calls: {num_api_calls} | Tokens: {total_prompt_tokens} in, {total_completion_tokens} out'
    )
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [final_verdict],
            'individual_answers': {t: all_answers.get(t) for t in agent_types},
            'votes': {t: all_votes.get(t) for t in agent_types},
            'final_vote': final_vote,
            'vote_distribution': dict(vote_counts),
            'label': ground_truth,
            'label_normalized': ground_truth_normalized,
            'correct': is_correct,
            'method': 'modality_ensemble',
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
    parser = argparse.ArgumentParser(
        description='Modality Ensemble (Text+Audio+Video majority voting) for Affective Detection'
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
    args = parser.parse_args()

    # Get dataset configuration
    config = get_dataset_config(args.dataset_name)

    # Set default dataset path if not specified
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    output_file = (
        args.output
        or f'modality_ensemble_{args.dataset_name}_{args.provider}.jsonl'
    )

    check_api_key(args.provider)
    litellm.set_verbose = False

    # Initialize tracker
    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )

    # Create 3 modality-specific agents
    text_agent = create_agent(
        'text', provider=args.provider, model=args.model, temperature=args.temperature
    )
    audio_agent = create_agent(
        'audio', provider=args.provider, model=args.model, temperature=args.temperature
    )
    video_agent = create_agent(
        'video', provider=args.provider, model=args.model, temperature=args.temperature
    )
    print(f'Dataset: {args.dataset_name} ({config.task_type})')
    print(f'Provider: {args.provider} | Model: {text_agent.model}')
    print(f'Agents: TextAgent + AudioAgent + VideoAgent -> Majority Vote')

    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    print(f'Total Test Cases: {len(test_list)}')

    # Load already processed keys for resume
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f'Resuming: {len(processed_keys)} already processed, skipping...')

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
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
                print(f'[ERROR] Failed to process {test_file}: {e}')
                print('[INFO] Skipping and continuing with next sample...')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary('Modality Ensemble (Text+Audio+Video)')
