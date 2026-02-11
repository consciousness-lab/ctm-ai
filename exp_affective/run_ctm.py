"""
CTM (Conscious Turing Machine) experiment

Use CTM framework for affective computing (humor detection, sarcasm detection, etc.)

Examples:
python run_ctm.py --dataset_name urfunny --max_workers 8
python run_ctm.py --dataset_name mustard --max_workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ctm_ai.ctms.ctm import ConsciousTuringMachine
from dataset_configs import get_dataset_config
from llm_utils import get_audio_path, get_muted_video_path, load_data

sys.path.append('..')


def load_data_local(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


file_lock = Lock()


def run_instance(
    test_file,
    dataset,
    dataset_name,
    ctm_name,
    output_file='ctm_results.jsonl',
):
    try:
        print(f'[{test_file}] Starting processing...')

        config = get_dataset_config(dataset_name)
        sample = dataset[test_file]

        ctm = ConsciousTuringMachine(ctm_name)
        target_sentence = config.get_text_field(sample)
        query = config.get_task_query()

        print(f'[{test_file}] Target sentence: {target_sentence[:100]}...')
        print(f'[{test_file}] Query: {query}')

        audio_path = get_audio_path(test_file, dataset_name)
        video_path = get_muted_video_path(test_file, dataset_name)

        if not os.path.exists(audio_path):
            print(f'[{test_file}] Audio not exist: {audio_path}')
            audio_path = None
        else:
            print(f'[{test_file}] Audio found: {audio_path}')

        if not os.path.exists(video_path):
            print(f'[{test_file}] Video not exist: {video_path}')
            video_path = None
        else:
            print(f'[{test_file}] Video found: {video_path}')

        print(f'[{test_file}] Calling CTM...')
        start_time = time.time()
        answer, weight_score, parsed_answer = ctm(
            query=query,
            text=target_sentence,
            video_path=video_path,
            audio_path=audio_path,
        )
        end_time = time.time()

        print(f'[{test_file}] CTM call completed in {end_time - start_time:.2f}s')
        print(f'[{test_file}] Answer: {answer}')
        print(f'[{test_file}] Parsed answer: {parsed_answer}')

        iteration_history = ctm.iteration_history
        num_iterations = len(iteration_history)
        winning_processors = [it['winning_processor'] for it in iteration_history]
        print(
            f'[{test_file}] Iterations: {num_iterations}, Winners: {winning_processors}'
        )

        label = config.get_label_field(sample)
        result = {
            test_file: {
                'answer': [answer],
                'parsed_answer': [parsed_answer],
                'weight_score': weight_score,
                'label': label,
                'num_iterations': num_iterations,
                'winning_processors': winning_processors,
            }
        }

        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f'[{test_file}] Result saved to {output_file}')
        return f'Successfully processed {test_file}'

    except Exception as e:
        error_msg = f'Error processing {test_file}: {str(e)}'
        print(f'[{test_file}] {error_msg}')
        import traceback

        traceback.print_exc()
        return error_msg


def run_parallel(
    dataset, dataset_name, ctm_name, max_workers=4, output_file='ctm_results.jsonl'
):
    test_list = list(dataset.keys())
    print(f'Total test samples: {len(test_list)}')
    print(f'Using {max_workers} workers')
    print(f'Output file: {output_file}')
    print('=' * 50)

    with open(output_file, 'w', encoding='utf-8'):
        pass

    start_time = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(
                run_instance, test_file, dataset, dataset_name, ctm_name, output_file
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
    print(f'Average time per sample: {total_time / len(test_list):.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTM for Affective Detection')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='urfunny',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: urfunny)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset JSON file (default: auto based on dataset_name)',
    )
    parser.add_argument(
        '--ctm_name',
        type=str,
        default=None,
        help='CTM name (default: auto based on dataset)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 16)',
    )
    args = parser.parse_args()

    # Get dataset configuration
    config = get_dataset_config(args.dataset_name)

    # Set default dataset path if not specified
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    # Set default CTM name
    if args.ctm_name is None:
        ctm_names = {
            'urfunny': 'urfunny_test',
            'mustard': 'sarcasm_ctm',
        }
        args.ctm_name = ctm_names.get(args.dataset_name, f'{args.dataset_name}_ctm')

    output_file = args.output or f'ctm_{args.dataset_name}.jsonl'

    print(f'Dataset: {args.dataset_name} ({config.task_type})')
    print(f'CTM Name: {args.ctm_name}')

    dataset = load_data(args.dataset)

    run_parallel(
        dataset,
        args.dataset_name,
        args.ctm_name,
        max_workers=args.max_workers,
        output_file=output_file,
    )
