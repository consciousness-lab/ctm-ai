"""
Test script: Verify unified framework setup

Checks:
1. Dataset configuration is correct
2. Data paths exist
3. llm_utils tools work properly
"""

import os
import sys

from dataset_configs import DATASET_CONFIGS, get_dataset_config
from llm_utils import (
    get_audio_path,
    get_full_video_path,
    get_muted_video_path,
    load_data,
)


def test_dataset_config(dataset_name: str):
    """Test dataset configuration"""
    print(f'\n{"=" * 60}')
    print(f'Testing dataset: {dataset_name}')
    print(f'{"=" * 60}')

    try:
        config = get_dataset_config(dataset_name)
        print(f'✓ Configuration loaded successfully')
        print(f'  - Name: {config.name}')
        print(f'  - Task type: {config.task_type}')

        # Test data paths
        data_paths = config.get_data_paths()
        print(f'\nData path check:')
        for key, path in data_paths.items():
            exists = os.path.exists(path)
            status = '✓' if exists else '✗'
            print(f'  {status} {key}: {path}')

        # Test default dataset path
        default_dataset = config.get_default_dataset_path()
        exists = os.path.exists(default_dataset)
        status = '✓' if exists else '✗'
        print(f'\nDefault dataset file:')
        print(f'  {status} {default_dataset}')

        # Load data and test first sample
        if exists:
            dataset = load_data(default_dataset)
            print(f'\nDataset statistics:')
            print(f'  - Number of samples: {len(dataset)}')

            # Test first sample
            first_key = list(dataset.keys())[0]
            first_sample = dataset[first_key]
            print(f'\nFirst sample test (ID: {first_key}):')

            # Test field extraction
            text_field = config.get_text_field(first_sample)
            print(f'  - Target text: {text_field[:80]}...')

            context_field = config.get_context_field(first_sample)
            print(f'  - Full context: {len(context_field)} characters')

            label = config.get_label_field(first_sample)
            print(f'  - Label: {label}')

            # Test file paths
            print(f'\nFile path test:')
            audio_path = get_audio_path(first_key, dataset_name)
            audio_exists = os.path.exists(audio_path)
            print(f'  {"✓" if audio_exists else "✗"} Audio: {audio_path}')

            video_path = get_muted_video_path(first_key, dataset_name)
            video_exists = os.path.exists(video_path)
            print(f'  {"✓" if video_exists else "✗"} Muted video: {video_path}')

            full_video_path = get_full_video_path(first_key, dataset_name)
            full_video_exists = os.path.exists(full_video_path)
            print(
                f'  {"✓" if full_video_exists else "✗"} Full video: {full_video_path}'
            )

        # Test prompts
        print(f'\nPrompts check:')
        task_query = config.get_task_query()
        print(f'  - Task query: {task_query}')

        system_prompt = config.get_system_prompt()
        print(f'  - System prompt: {len(system_prompt)} characters')

        debate_prompts = config.get_debate_prompts()
        print(f'  - Debate prompts: {len(debate_prompts)} items')

        query_aug_prompts = config.get_query_aug_prompts()
        print(f'  - Query aug prompts: {len(query_aug_prompts)} items')

        print(f'\n✓ Dataset {dataset_name} test passed!')
        return True

    except Exception as e:
        print(f'\n✗ Dataset {dataset_name} test failed!')
        print(f'Error: {e}')
        import traceback

        traceback.print_exc()
        return False


def main():
    print('Unified Affective Computing Framework - Setup Test')
    print('=' * 60)

    # Test all registered datasets
    results = {}
    for dataset_name in DATASET_CONFIGS.keys():
        results[dataset_name] = test_dataset_config(dataset_name)

    # Summary
    print(f'\n{"=" * 60}')
    print('Test Summary')
    print(f'{"=" * 60}')
    for dataset_name, passed in results.items():
        status = '✓ Passed' if passed else '✗ Failed'
        print(f'{status} - {dataset_name}')

    all_passed = all(results.values())
    if all_passed:
        print(f'\n✓ All tests passed! Framework is ready.')
        return 0
    else:
        print(f'\n✗ Some tests failed, please check configuration.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
