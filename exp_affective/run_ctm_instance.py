"""
CTM (Conscious Turing Machine) - Single Instance Debug Runner

Script for debugging a single sample, useful for troubleshooting.

Examples:
python run_ctm_instance.py --instance_id 2_380 --dataset_name mustard
python run_ctm_instance.py --instance_id 1_1 --dataset_name urfunny
"""

import argparse
import json
import os
import sys
import time

from ctm_ai.ctms.ctm import ConsciousTuringMachine
from dataset_configs import get_dataset_config
from llm_utils import get_audio_path, get_muted_video_path, load_data

sys.path.append('..')


def run_single_instance(
    instance_id,
    dataset,
    dataset_name,
    ctm_name,
    output_file=None,
    verbose=True,
):
    """Run a single instance for debugging."""

    if instance_id not in dataset:
        print(f'Error: instance_id "{instance_id}" not found in dataset')
        print(f'Available instance_id examples (first 10):')
        for i, key in enumerate(list(dataset.keys())[:10]):
            print(f'  - {key}')
        print(f'...')
        print(f'Total {len(dataset)} samples')
        return None

    config = get_dataset_config(dataset_name)
    sample = dataset[instance_id]

    if verbose:
        print('=' * 60)
        print(f'Instance ID: {instance_id}')
        print(f'Dataset: {dataset_name}')
        print(f'CTM Name: {ctm_name}')
        print('=' * 60)
        print()

    # Show sample info
    target_sentence = config.get_text_field(sample)
    query = config.get_task_query()
    label = config.get_label_field(sample)

    if verbose:
        print('[Sample Info]')
        print(f'  Text: {target_sentence}')
        print(f'  Label: {label}')
        print(f'  Query: {query}')
        print()

    # Check media files
    audio_path = get_audio_path(instance_id, dataset_name)
    video_path = get_muted_video_path(instance_id, dataset_name)

    if verbose:
        print('[Media Files]')

    if not os.path.exists(audio_path):
        if verbose:
            print(f'  Audio: not found ({audio_path})')
        audio_path = None
    else:
        if verbose:
            print(f'  Audio: {audio_path}')

    if not os.path.exists(video_path):
        if verbose:
            print(f'  Video: not found ({video_path})')
        video_path = None
    else:
        if verbose:
            print(f'  Video: {video_path}')

    if verbose:
        print()
        print('[Starting CTM call]')
        print('-' * 60)

    # Initialize CTM
    ctm = ConsciousTuringMachine(ctm_name)

    # Call CTM
    start_time = time.time()
    answer, weight_score, parsed_answer = ctm(
        query=query,
        text=target_sentence,
        video_path=video_path,
        audio_path=audio_path,
    )
    end_time = time.time()

    if verbose:
        print('-' * 60)
        print()
        print('[CTM Result]')
        print(f'  Time: {end_time - start_time:.2f}s')
        print(f'  Answer: {answer}')
        print(f'  Parsed Answer: {parsed_answer}')
        print(f'  Weight Score: {weight_score}')
        print(f'  Label (Ground Truth): {label}')
        print(f'  Correct: {"✓" if parsed_answer == label else "✗"}')
        print()

    result = {
        instance_id: {
            'answer': [answer],
            'parsed_answer': [parsed_answer],
            'weight_score': weight_score,
            'label': label,
        }
    }

    # Save result to file (optional)
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        if verbose:
            print(f'Result saved to: {output_file}')

    return result


def list_instances(dataset, limit=20):
    """List all instance_ids in the dataset."""
    print(f'Dataset has {len(dataset)} samples')
    print(f'Instance IDs (first {limit}):')
    for i, key in enumerate(list(dataset.keys())[:limit]):
        print(f'  {i + 1}. {key}')
    if len(dataset) > limit:
        print(f'  ... {len(dataset) - limit} more')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CTM Single Instance Debug Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ctm_instance.py --instance_id 2_380 --dataset_name mustard
  python run_ctm_instance.py --instance_id 1_1 --dataset_name urfunny
  python run_ctm_instance.py --list --dataset_name mustard
        """,
    )
    parser.add_argument(
        '--instance_id',
        type=str,
        default=None,
        help='Instance ID to run, e.g. "2_380"',
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='mustard',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: mustard)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset JSON path (default: auto from dataset_name)',
    )
    parser.add_argument(
        '--ctm_name',
        type=str,
        default=None,
        help='CTM config name (default: auto from dataset)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL path (optional, omit to skip saving)',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all instance_ids in the dataset',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity',
    )
    args = parser.parse_args()

    # Get dataset config
    config = get_dataset_config(args.dataset_name)

    # Set default dataset path
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    # Set default CTM name
    if args.ctm_name is None:
        ctm_names = {
            'urfunny': 'urfunny_test',
            'mustard': 'sarcasm_ctm',
        }
        args.ctm_name = ctm_names.get(args.dataset_name, f'{args.dataset_name}_ctm')

    # Load dataset
    dataset = load_data(args.dataset)

    # If list mode
    if args.list:
        list_instances(dataset)
        sys.exit(0)

    # Check if instance_id provided
    if args.instance_id is None:
        print('Error: please provide --instance_id')
        print('Use --list to see available instance_ids')
        sys.exit(1)

    # Run single instance
    run_single_instance(
        instance_id=args.instance_id,
        dataset=dataset,
        dataset_name=args.dataset_name,
        ctm_name=args.ctm_name,
        output_file=args.output,
        verbose=not args.quiet,
    )
