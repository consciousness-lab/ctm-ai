"""Run CTM only on specific target keys (skip everything else)."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from dataset_configs import get_dataset_config
from llm_utils import get_audio_path, get_muted_video_path, load_data

from ctm_ai.ctms.ctm import ConsciousTuringMachine

sys.path.append('..')

file_lock = Lock()


def run_instance(test_file, dataset, dataset_name, ctm_name, output_file):
    try:
        config = get_dataset_config(dataset_name)
        sample = dataset[test_file]
        ctm = ConsciousTuringMachine(ctm_name)
        target_sentence = config.get_text_field(sample)
        query = config.get_task_query()

        audio_path = get_audio_path(test_file, dataset_name)
        video_path = get_muted_video_path(test_file, dataset_name)
        if not os.path.exists(audio_path):
            audio_path = None
        if not os.path.exists(video_path):
            video_path = None

        start_time = time.time()
        answer, weight_score, parsed_answer = ctm(
            query=query,
            text=target_sentence,
            video_path=video_path,
            audio_path=audio_path,
            instance_id=test_file,
        )
        elapsed = time.time() - start_time

        iteration_history = ctm.iteration_history
        num_iterations = len(iteration_history)
        winning_processors = [it['winning_processor'] for it in iteration_history]

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

        print(f'[{test_file}] Done in {elapsed:.0f}s')
        return f'{test_file} done'
    except Exception as e:
        print(f'[{test_file}] Error: {e}')
        import traceback
        traceback.print_exc()
        return f'{test_file} error'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mustard')
    parser.add_argument('--ctm_name', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target_keys', type=str, required=True, help='JSON file with list of target keys')
    parser.add_argument('--max_workers', type=int, default=5)
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    dataset = load_data(config.get_default_dataset_path())

    with open(args.target_keys) as f:
        target_keys = set(json.load(f))

    # Skip already done
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                done.update(json.loads(line).keys())

    remaining = [k for k in target_keys if k not in done]
    print(f'Target: {len(target_keys)}, Done: {len(target_keys & done)}, Remaining: {len(remaining)}')

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_instance, k, dataset, args.dataset_name, args.ctm_name, args.output): k
            for k in remaining
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            print(f'Progress: {done_count}/{len(remaining)} - {future.result()}')

    print(f'\nAll done. {done_count} processed.')
