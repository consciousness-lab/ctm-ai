"""
Baseline experiment using multimodal input.

Sends full video (with audio) + text (target sentence) in a single call per sample.
This is the simplest baseline: no debate, no augmentation, no voting.

Examples:
python run_pure_llm.py --dataset_name urfunny --provider gemini --output pure_llm_urfunny_gemini.jsonl
python run_pure_llm.py --dataset_name mustard --provider gemini --output pure_llm_mustard_gemini.jsonl
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    check_api_key,
    create_agent,
    get_full_video_path,
    load_data,
    load_processed_keys,
    save_result_to_jsonl,
)

sys.path.append('..')

file_lock = Lock()


def run_instance(test_file, dataset, dataset_name, agent, output_file='baseline.jsonl'):
    try:
        print(f'[{test_file}] Starting processing...')

        config = get_dataset_config(dataset_name)
        sample = dataset[test_file]

        target_sentence = config.get_text_field(sample)
        system_prompt = config.get_system_prompt()

        # Prepare query with system prompt and target sentence
        query = f"{system_prompt}\n\ntarget text: '{target_sentence}'"

        # Full video (with audio)
        video_path = get_full_video_path(test_file, dataset_name)

        if not os.path.exists(video_path):
            print(f'[{test_file}] Video not exist: {video_path}')
            return f'Video file not found for {test_file}'

        print(f'[{test_file}] Target sentence: {target_sentence[:100]}...')
        print(f'[{test_file}] Video found: {video_path}')

        print(f'[{test_file}] Calling LLM...')
        start_time = time.time()

        # Use MultimodalAgent: full video (with audio) + query
        answer, usage = agent.call(
            query,
            video_path=video_path,
        )

        end_time = time.time()

        print(f'[{test_file}] LLM call completed in {end_time - start_time:.2f}s')
        print(f'[{test_file}] Answer: {answer}')

        label = config.get_label_field(sample)
        result = {
            test_file: {
                'answer': [answer],
                'label': label,
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                },
                'latency': end_time - start_time,
            }
        }

        with file_lock:
            save_result_to_jsonl(result, output_file)

        print(f'[{test_file}] Result saved to {output_file}')
        return f'Successfully processed {test_file}'

    except Exception as e:
        error_msg = f'Error processing {test_file}: {str(e)}'
        print(f'[{test_file}] {error_msg}')
        import traceback

        traceback.print_exc()
        return error_msg


def run_parallel(agent, dataset, dataset_name, output_file, max_workers=4):
    test_list = list(dataset.keys())
    print(f'Total test samples: {len(test_list)}')
    print(f'Using {max_workers} workers')
    print(f'Output file: {output_file}')
    print('=' * 50)

    # Load already processed keys for resume
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f'Resuming: {len(processed_keys)} already processed, skipping...')
        test_list = [t for t in test_list if t not in processed_keys]
        print(f'Remaining: {len(test_list)} samples')

    if not test_list:
        print('All samples already processed!')
        return

    start_time = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(
                run_instance, test_file, dataset, dataset_name, agent, output_file
            ): test_file
            for test_file in test_list
        }

        for future in as_completed(future_to_test):
            test_file = future_to_test[future]
            completed_count += 1

            try:
                result = future.result()
                print(f'Progress: {completed_count}/{len(test_list)} - {result}')
            except Exception as exc:
                print(f'Error processing {test_file}: {exc}')

    end_time = time.time()
    total_time = end_time - start_time
    print('=' * 50)
    print(f'Total processing time: {total_time:.2f} seconds')
    if test_list:
        print(f'Average time per sample: {total_time / len(test_list):.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Baseline Multimodal Affective Detection'
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
        '--max_workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 8)',
    )
    args = parser.parse_args()

    # Get dataset configuration
    config = get_dataset_config(args.dataset_name)

    # Set default dataset path if not specified
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    output_file = args.output or f'baseline_{args.dataset_name}_{args.provider}.jsonl'

    check_api_key(args.provider)
    litellm.set_verbose = False

    # Create multimodal agent (full video with audio + text)
    agent = create_agent(
        'multimodal',
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    print(f'Dataset: {args.dataset_name} ({config.task_type})')
    print(f'Provider: {args.provider} | Model: {agent.model}')

    dataset = load_data(args.dataset)

    run_parallel(
        agent, dataset, args.dataset_name, output_file, max_workers=args.max_workers
    )
