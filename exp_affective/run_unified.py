"""
Unified baseline: single MultimodalAgent call per sample.

Sends full video (with audio) + text in one call.
This is the simplest baseline: no debate, no augmentation, no voting.

Examples:
python run_unified.py --dataset_name urfunny --provider qwen --output unified_urfunny_qwen.jsonl
python run_unified.py --dataset_name mustard --provider gemini --output unified_mustard_gemini.jsonl
"""

import argparse
import sys
import time

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


def run_instance(test_file, dataset, dataset_name, agent, tracker, output_file):
    """Process one sample: load inputs -> call agent -> save result."""
    start_time = time.time()

    # Step 1: Load inputs
    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    system_prompt = inputs['system_prompt']
    label = inputs['label']
    full_video_path = inputs['full_video_path']

    query = f"{system_prompt}\n\ntarget text: '{target_sentence}'"

    print(f'[{test_file}] target: {target_sentence[:80]}...')

    # Step 2: Run logic — single multimodal call
    answer, usage = agent.call(query, video_path=full_video_path)

    end_time = time.time()
    duration = end_time - start_time
    tracker.add(
        duration,
        usage.get('prompt_tokens', 0),
        usage.get('completion_tokens', 0),
        1,
    )

    print(f'[{test_file}] answer: {answer[:80] if answer else "None"}... ({duration:.1f}s)')

    # Step 3: Save result
    result = {
        test_file: {
            'answer': [answer],
            'label': label,
            'label_normalized': normalize_label(label),
            'method': 'unified',
            'usage': {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'api_calls': 1,
            },
            'latency': duration,
        }
    }
    save_result_to_jsonl(result, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Baseline')
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
    output_file = args.output or f'unified_{args.dataset_name}_{args.provider}.jsonl'

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
    print(f'Dataset: {args.dataset_name} | Provider: {args.provider} | Model: {agent.model}')

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
                run_instance(test_file, dataset, args.dataset_name, agent, tracker, output_file)
            except Exception as e:
                print(f'[ERROR] {test_file}: {e}')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary('Unified Baseline')
