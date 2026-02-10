"""
Multimodal Voting experiment with N votes using multimodal input.

Each vote sends the full video (with audio) + target text.
N votes run in parallel, then majority vote determines the final answer.

Examples:
python run_voting.py --dataset_name urfunny --provider gemini --n_votes 3 --output voting_urfunny_gemini.jsonl
python run_voting.py --dataset_name mustard --provider gemini --n_votes 3 --output voting_mustard_gemini.jsonl
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
    add_common_args,
    check_api_key,
    create_agent,
    get_full_video_path,
    load_data,
    load_processed_keys,
    normalize_label,
    save_result_to_jsonl,
)

sys.path.append('..')

# Number of votes
N_VOTES = 3

# Higher temperature for diversity in voting
TEMPERATURE = 1.0

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


def validate_answer_format(answer):
    """Validate if the answer follows the expected format"""
    if answer is None or not answer.strip():
        return False, 'Empty answer'

    lines = answer.strip().split('\n')
    first_line = lines[0].strip().lower()
    first_word = first_line.split()[0] if first_line.split() else ''

    if first_word not in ['yes', 'no']:
        return False, f"First word is '{first_word}', expected 'Yes' or 'No'"

    if len(lines) < 2:
        return False, 'Missing reasoning (expected multiple lines)'

    reasoning = '\n'.join(lines[1:]).strip()
    if len(reasoning) < 10:
        return False, 'Reasoning too short'

    return True, 'Valid format'


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
    multimodal_agent,
    tracker,
    output_file='voting.jsonl',
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

    # Full video path (contains both video and audio)
    full_video_path = get_full_video_path(test_file, dataset_name)

    def single_vote(vote_idx):
        """Execute a single vote with multimodal input"""
        answer, usage = multimodal_agent.call(query, video_path=full_video_path)
        return vote_idx, answer, usage

    print(f'--- Voting ({N_VOTES} votes) for {test_file} ---')

    all_answers = []
    all_votes = []
    format_violations = 0

    # Use ThreadPoolExecutor for parallel voting
    with ThreadPoolExecutor(max_workers=N_VOTES) as executor:
        futures = {executor.submit(single_vote, i): i for i in range(N_VOTES)}

        for future in as_completed(futures):
            vote_idx = futures[future]
            try:
                idx, answer, usage = future.result()
                all_answers.append(answer)

                # Validate format
                is_valid, validation_msg = validate_answer_format(answer)
                if not is_valid:
                    format_violations += 1
                    print(f'  Vote {idx + 1}: [FORMAT ERROR] {validation_msg}')

                vote = extract_vote(answer)
                all_votes.append(vote)

                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                num_api_calls += 1

                # Show first 100 chars of reasoning
                answer_preview = (
                    answer.strip().replace('\n', ' ')[:100] if answer else 'None'
                )
                format_indicator = '✓' if is_valid else '✗'
                print(
                    f'  Vote {idx + 1}: {vote:<7} {format_indicator} | {answer_preview}...'
                )

            except Exception as exc:
                print(f'  Vote {vote_idx + 1}: ERROR   | Exception: {exc}')
                all_answers.append('Error')
                all_votes.append('Unknown')
                format_violations += 1

    # Majority voting
    final_vote = majority_vote(all_votes)
    vote_counts = Counter(all_votes)
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
    if format_violations > 0:
        print(
            f'Format Issues: {format_violations}/{N_VOTES} votes had format violations'
        )
    print(
        f'Time: {duration:.2f}s | API Calls: {num_api_calls} | Tokens: {total_prompt_tokens} in, {total_completion_tokens} out'
    )
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [final_verdict],
            'individual_answers': all_answers,
            'votes': all_votes,
            'final_vote': final_vote,
            'vote_distribution': dict(vote_counts),
            'format_violations': format_violations,
            'label': ground_truth,
            'label_normalized': ground_truth_normalized,
            'correct': is_correct,
            'method': f'voting_n{N_VOTES}',
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
        description='Multimodal Voting for Affective Detection'
    )
    add_common_args(parser)
    parser.add_argument(
        '--n_votes',
        type=int,
        default=3,
        help='Number of votes (default: 3)',
    )
    args = parser.parse_args()

    N_VOTES = args.n_votes
    TEMPERATURE = args.temperature

    # Get dataset configuration
    config = get_dataset_config(args.dataset_name)

    # Set default dataset path if not specified
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    output_file = args.output or f'voting_{args.dataset_name}_{args.provider}.jsonl'

    check_api_key(args.provider)
    litellm.set_verbose = False

    # Initialize tracker
    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )

    # Create multimodal agent
    multimodal_agent = create_agent(
        'multimodal', provider=args.provider, model=args.model, temperature=TEMPERATURE
    )
    print(f'Dataset: {args.dataset_name} ({config.task_type})')
    print(f'Provider: {args.provider} | Model: {multimodal_agent.model}')

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
                    multimodal_agent,
                    tracker,
                    output_file,
                )
            except Exception as e:
                print(f'[ERROR] Failed to process {test_file}: {e}')
                print('[INFO] Skipping and continuing with next sample...')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary(f'Voting N={N_VOTES}')
