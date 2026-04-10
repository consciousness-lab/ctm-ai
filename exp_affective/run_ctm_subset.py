"""
Run CTM on a subset of samples and compute accuracy.
Usage:
    python run_ctm_subset.py --dataset_name urfunny --ctm_name urfunny_test_qwen --subset data/urfunny/data_raw/urfunny_subset_20.json --output results_baseline.jsonl
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


def run_subset(dataset, dataset_name, ctm_name, output_file, max_workers=1):
    config = get_dataset_config(dataset_name)
    test_list = list(dataset.keys())
    print(f'Total samples: {len(test_list)}')
    print(f'CTM: {ctm_name}')
    print(f'Output: {output_file}')
    print('=' * 60)

    # Load CTM config to check for custom task_query
    from ctm_ai.configs import ConsciousTuringMachineConfig
    ctm_cfg = ConsciousTuringMachineConfig.from_ctm(ctm_name)
    custom_query = getattr(ctm_cfg, 'task_query', None)
    if custom_query:
        print(f'Using custom query: {custom_query[:80]}...')

    # Load existing results for resume support
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    done_ids.update(d.keys())
                except Exception:
                    pass
    if done_ids:
        print(f'Resuming: {len(done_ids)} samples already done, skipping them.')

    results = []
    correct = 0
    total = 0
    total_time = 0

    for i, test_file in enumerate(test_list):
        if test_file in done_ids:
            continue
        sample = dataset[test_file]
        label = config.get_label_field(sample)
        # Use full context (context_sentences + punchline) instead of just punchline
        target_sentence = config.get_context_field(sample)
        query = custom_query or config.get_task_query()

        audio_path = get_audio_path(test_file, dataset_name)
        video_path = get_muted_video_path(test_file, dataset_name)

        if not os.path.exists(audio_path):
            audio_path = None
        if not os.path.exists(video_path):
            video_path = None

        print(f'\n[{i+1}/{len(test_list)}] ID={test_file} label={label}')
        print(f'  Text: {target_sentence[:80]}...')

        ctm = ConsciousTuringMachine(ctm_name)
        start = time.time()
        try:
            answer, weight_score, parsed_answer = ctm(
                query=query,
                text=target_sentence,
                video_path=video_path,
                audio_path=audio_path,
                instance_id=test_file,
            )
        except Exception as e:
            print(f'  ERROR: {e}')
            answer = 'ERROR'
            weight_score = 0
            parsed_answer = 'ERROR'
        elapsed = time.time() - start
        total_time += elapsed

        # Parse Yes/No from parsed_answer
        pred = None
        if isinstance(parsed_answer, str):
            pa_lower = parsed_answer.strip().lower()
            if pa_lower.startswith('yes'):
                pred = 1
            elif pa_lower.startswith('no'):
                pred = 0
            else:
                # Fallback: search for humor-related keywords
                if 'is humorous' in pa_lower or 'is being humorous' in pa_lower or 'using humor' in pa_lower:
                    pred = 1
                elif 'not humorous' in pa_lower or 'not being humorous' in pa_lower or 'no humor' in pa_lower:
                    pred = 0

        is_correct = (pred == label) if pred is not None else False
        if is_correct:
            correct += 1
        total += 1

        print(f'  Parsed: {parsed_answer[:100] if isinstance(parsed_answer, str) else parsed_answer}')
        print(f'  Pred={pred} Label={label} Correct={is_correct} Time={elapsed:.1f}s')

        result = {
            test_file: {
                'answer': answer[:200] if isinstance(answer, str) else answer,
                'parsed_answer': parsed_answer[:200] if isinstance(parsed_answer, str) else parsed_answer,
                'weight_score': weight_score,
                'label': label,
                'pred': pred,
                'correct': is_correct,
                'time': round(elapsed, 1),
            }
        }
        results.append(result)

        with open(output_file, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        acc_so_far = correct / total * 100
        print(f'  Running accuracy: {correct}/{total} = {acc_so_far:.1f}%')

    print('\n' + '=' * 60)
    print(f'FINAL RESULTS')
    print(f'Accuracy: {correct}/{total} = {correct/total*100:.1f}%')
    print(f'Total time: {total_time:.1f}s')
    print(f'Avg time per sample: {total_time/total:.1f}s')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='urfunny')
    parser.add_argument('--ctm_name', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True, help='Path to subset JSON')
    parser.add_argument('--output', type=str, default='results.jsonl')
    args = parser.parse_args()

    dataset = load_data(args.subset)
    run_subset(dataset, args.dataset_name, args.ctm_name, args.output)
