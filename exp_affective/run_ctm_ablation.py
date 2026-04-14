"""
CTM Ablation runner for affective benchmarks (MUSTARD / URFunny).

Uses AblationCTM which supports:
  - --output_threshold          override config output_threshold
  - --link_form_threshold       override hardcoded 0.8 link-add threshold
  - --link_break_threshold      override hardcoded 0.2 link-break threshold
  - --ablation                  component ablation flags
  - --modality                  single-modality ablation
  - --run_output_thresholds     preset sweep
  - --run_link_thresholds       preset sweep
  - --resume                    continue from an existing output file

Results folder structure:
  <results_dir>/
      ctm_<dataset>_<tag>.jsonl
      detailed_info_<dataset>_<tag>/<instance>.json

If --results_dir is not set, files are written to the current directory
and trajectories to detailed_info/.

Examples:
  python run_ctm_ablation.py --dataset_name urfunny --ctm_name urfunny_test_gemini_v28 \\
      --output_threshold 2.1 --results_dir urfunny_v28_ablation

  python run_ctm_ablation.py --dataset_name urfunny --ctm_name urfunny_test_gemini_v28 \\
      --link_form_threshold 0.5 --link_break_threshold 0.3 \\
      --results_dir urfunny_v28_ablation --max_workers 4
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Ensure we import ctm_ai from this repo, not a pip-installed copy.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset_configs import get_dataset_config
from llm_utils import get_audio_path, get_muted_video_path, load_data

from ctm_ai.ctms.ctm_ablation import AblationCTM

ABLATION_CHOICES = [
    'no_fusion',
    'no_uptree',
    'no_broadcast',
    'no_link_form',
    'single_iter',
]

file_lock = Lock()


def parse_ablation_flags(ablation_list):
    flags = {
        'enable_fusion': True,
        'enable_uptree_competition': True,
        'enable_downtree_broadcast': True,
        'enable_link_form': True,
        'enable_iteration': True,
    }
    mapping = {
        'no_fusion': 'enable_fusion',
        'no_uptree': 'enable_uptree_competition',
        'no_broadcast': 'enable_downtree_broadcast',
        'no_link_form': 'enable_link_form',
        'single_iter': 'enable_iteration',
    }
    for ablation in ablation_list:
        flags[mapping[ablation]] = False
    return flags


def make_ablation_tag(ablation_list):
    if not ablation_list:
        return 'full_ctm'
    return '_'.join(sorted(ablation_list))


def run_instance(
    test_file,
    dataset,
    dataset_name,
    ctm_name,
    ablation_flags,
    ablation_tag,
    output_file,
    modality=None,
    threshold_kwargs=None,
    detailed_log_dir=None,
    task_query_override=None,
):
    try:
        print(f'[{test_file}] Starting processing ({ablation_tag})...')

        config = get_dataset_config(dataset_name)
        sample = dataset[test_file]

        ctm = AblationCTM(
            ctm_name,
            **ablation_flags,
            **(threshold_kwargs or {}),
            detailed_log_dir=detailed_log_dir,
        )

        # Honor task_query override if provided (e.g. from config file)
        if task_query_override:
            query = task_query_override
        else:
            query = config.get_task_query()

        target_sentence = config.get_text_field(sample)
        audio_path = get_audio_path(test_file, dataset_name)
        video_path = get_muted_video_path(test_file, dataset_name)

        if not os.path.exists(audio_path):
            audio_path = None
        if not os.path.exists(video_path):
            video_path = None

        if modality == 'video':
            ctm.remove_processor('audio_processor')
            ctm.remove_processor('language_processor')
        elif modality == 'audio':
            ctm.remove_processor('video_processor')
            ctm.remove_processor('language_processor')
        elif modality == 'text':
            ctm.remove_processor('video_processor')
            ctm.remove_processor('audio_processor')

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

        total_links_added = ctm._total_links_added
        total_links_broken = ctm._total_links_broken
        ctm_usage = ctm.get_usage_stats()
        parse_usage = ctm.get_parse_usage_stats()
        links_per_iter = [
            {
                'iter': it['iteration'],
                'added': it.get('links_added', 0),
                'broken': it.get('links_broken', 0),
            }
            for it in iteration_history
        ]

        # Cost calculation (Gemini 2.5 Flash Lite pricing)
        all_prompt = ctm_usage['prompt_tokens'] + parse_usage['prompt_tokens']
        all_completion = ctm_usage['completion_tokens'] + parse_usage['completion_tokens']
        input_cost = all_prompt * 0.075 / 1e6
        output_cost = all_completion * 0.30 / 1e6
        total_cost = input_cost + output_cost

        print(
            f'[{test_file}] Done in {elapsed:.1f}s | '
            f'iters={num_iterations} | links_added={total_links_added} | '
            f'api_calls={ctm_usage["api_calls"]}+1parse | '
            f'tokens={ctm_usage["total_tokens"]}+{parse_usage["total_tokens"]}parse | '
            f'cost=${total_cost:.4f} | '
            f"parsed={parsed_answer[:50]}"
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
                'ablation': ablation_tag,
                'latency_seconds': round(elapsed, 2),
                'ctm_api_calls': ctm_usage['api_calls'],
                'ctm_prompt_tokens': ctm_usage['prompt_tokens'],
                'ctm_completion_tokens': ctm_usage['completion_tokens'],
                'ctm_total_tokens': ctm_usage['total_tokens'],
                'parse_api_calls': parse_usage['api_calls'],
                'parse_prompt_tokens': parse_usage['prompt_tokens'],
                'parse_completion_tokens': parse_usage['completion_tokens'],
                'parse_total_tokens': parse_usage['total_tokens'],
                'total_cost_usd': round(total_cost, 6),
                'total_links_added': total_links_added,
                'total_links_broken': total_links_broken,
                'links_per_iter': links_per_iter,
            }
        }

        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        return f'OK {test_file}'

    except Exception as e:
        print(f'[{test_file}] ERROR: {e}')
        import traceback

        traceback.print_exc()
        return f'FAIL {test_file}: {e}'


def _load_completed_ids(output_file):
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                completed.update(result.keys())
            except json.JSONDecodeError:
                continue
    return completed


def _load_config_task_query(ctm_name):
    """Read task_query from the ctm config JSON, if present."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'ctm_conf', f'{ctm_name}_config.json'
    )
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg.get('task_query')
    except Exception:
        return None


def run_ablation(
    dataset,
    dataset_name,
    ctm_name,
    ablation_list,
    max_workers=8,
    output_file=None,
    results_dir=None,
    resume=False,
    modality=None,
    threshold_kwargs=None,
):
    ablation_flags = parse_ablation_flags(ablation_list)
    ablation_tag = make_ablation_tag(ablation_list)

    tk = threshold_kwargs or {}
    if tk.get('link_form_threshold') is not None or tk.get('link_break_threshold') is not None:
        lf = tk.get('link_form_threshold', 0.8)
        lb = tk.get('link_break_threshold', 0.2)
        ablation_tag = f'{ablation_tag}_lf{lf}_lb{lb}'
    if tk.get('output_threshold_override') is not None:
        ablation_tag = f'{ablation_tag}_ot{tk["output_threshold_override"]}'
    if tk.get('max_iter_override') is not None:
        ablation_tag = f'{ablation_tag}_iter{tk["max_iter_override"]}'

    if modality:
        ablation_tag = f'{ablation_tag}_only_{modality}'

    # Resolve output paths
    results_dir_abs = None
    if results_dir:
        results_dir_abs = os.path.abspath(results_dir)
        os.makedirs(results_dir_abs, exist_ok=True)

    if output_file is None:
        filename = f'ctm_{dataset_name}_{ctm_name}_{ablation_tag}.jsonl'
        if results_dir_abs:
            output_file = os.path.join(results_dir_abs, filename)
        else:
            output_file = filename

    # Trajectory directory: <results_dir>/detailed_info_<ctm_name>_<ablation_tag>
    if results_dir_abs:
        detailed_log_dir = os.path.join(
            results_dir_abs, f'detailed_info_{ctm_name}_{ablation_tag}'
        )
    else:
        detailed_log_dir = f'detailed_info_{ctm_name}_{ablation_tag}'
    os.makedirs(detailed_log_dir, exist_ok=True)

    task_query_override = _load_config_task_query(ctm_name)

    test_list = list(dataset.keys())

    if resume:
        done_ids = _load_completed_ids(output_file)
        test_list = [t for t in test_list if t not in done_ids]
        print(f'Resuming: {len(done_ids)} already done, {len(test_list)} remaining')
    else:
        with open(output_file, 'w', encoding='utf-8'):
            pass

    print('=' * 60)
    print(f'Ablation: {ablation_tag}')
    print(f'CTM config: {ctm_name}')
    print(f'Flags: {ablation_flags}')
    if tk:
        print(f'Thresholds: {tk}')
    if task_query_override:
        print(f'task_query (from config): {task_query_override[:80]}')
    print(f'Samples: {len(test_list)} | Workers: {max_workers}')
    print(f'Output: {output_file}')
    print(f'Trajectories: {detailed_log_dir}')
    print('=' * 60)

    if not test_list:
        print(f'Ablation [{ablation_tag}] nothing to do — all samples complete.')
        return output_file

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(
                run_instance,
                test_file,
                dataset,
                dataset_name,
                ctm_name,
                ablation_flags,
                ablation_tag,
                output_file,
                modality,
                tk,
                detailed_log_dir,
                task_query_override,
            ): test_file
            for test_file in test_list
        }

        for future in as_completed(future_to_test):
            completed += 1
            test_file = future_to_test[future]
            try:
                result = future.result()
                print(f'Progress: {completed}/{len(test_list)} - {result}')
            except Exception as exc:
                print(f'Error {test_file}: {exc}')

    total_time = time.time() - start_time
    print(
        f'Ablation [{ablation_tag}] done: {total_time:.1f}s total, '
        f'{total_time / max(len(test_list), 1):.1f}s/sample'
    )
    print()

    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CTM Ablation for Affective Detection'
    )
    parser.add_argument(
        '--dataset_name', type=str, default='mustard',
        choices=['urfunny', 'mustard'],
    )
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ctm_name', type=str, default=None)
    parser.add_argument(
        '--ablation', nargs='+', choices=ABLATION_CHOICES, default=[],
        help='Ablation(s) to apply. Omit for full CTM.',
    )
    parser.add_argument('--run_all', action='store_true',
        help='Run all 5 single-component ablations sequentially.')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None,
        help='Root directory for output files and trajectories.')
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--modality', type=str, choices=['video', 'audio', 'text'], default=None,
    )
    parser.add_argument('--run_all_modalities', action='store_true')

    # Threshold ablation arguments
    parser.add_argument('--link_form_threshold', type=float, default=None)
    parser.add_argument('--link_break_threshold', type=float, default=None)
    parser.add_argument('--output_threshold', type=float, default=None)
    parser.add_argument('--max_iter', type=int, default=None,
        help='Override max_iter_num in config.')
    parser.add_argument('--run_link_thresholds', action='store_true')
    parser.add_argument('--run_output_thresholds', action='store_true')
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)

    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    if args.ctm_name is None:
        ctm_names = {'urfunny': 'urfunny_test', 'mustard': 'sarcasm_ctm'}
        args.ctm_name = ctm_names.get(args.dataset_name, f'{args.dataset_name}_ctm')

    print(f'Dataset: {args.dataset_name} ({config.task_type})')
    print(f'CTM Name: {args.ctm_name}')

    dataset = load_data(args.dataset)

    threshold_kwargs = {}
    if args.link_form_threshold is not None:
        threshold_kwargs['link_form_threshold'] = args.link_form_threshold
    if args.link_break_threshold is not None:
        threshold_kwargs['link_break_threshold'] = args.link_break_threshold
    if args.output_threshold is not None:
        threshold_kwargs['output_threshold_override'] = args.output_threshold
    if args.max_iter is not None:
        threshold_kwargs['max_iter_override'] = args.max_iter

    LINK_THRESHOLD_PRESETS = [
        {'link_form_threshold': 0.3, 'link_break_threshold': 0.5},
        {'link_form_threshold': 0.5, 'link_break_threshold': 0.3},
        {'link_form_threshold': 0.8, 'link_break_threshold': 0.2},
        {'link_form_threshold': 0.9, 'link_break_threshold': 0.1},
        {'link_form_threshold': 0.95, 'link_break_threshold': 0.05},
    ]
    OUTPUT_THRESHOLD_PRESETS = [1.2, 1.5, 1.8, 2.1, 2.5, 3.0]

    def _run(tk):
        run_ablation(
            dataset,
            args.dataset_name,
            args.ctm_name,
            args.ablation,
            max_workers=args.max_workers,
            output_file=args.output,
            results_dir=args.results_dir,
            resume=args.resume,
            modality=args.modality,
            threshold_kwargs=tk or None,
        )

    if args.run_link_thresholds:
        for preset in LINK_THRESHOLD_PRESETS:
            _run(preset)
    elif args.run_output_thresholds:
        for ot in OUTPUT_THRESHOLD_PRESETS:
            _run({'output_threshold_override': ot})
    elif args.run_all_modalities:
        for mod in ['video', 'audio', 'text']:
            run_ablation(
                dataset,
                args.dataset_name,
                args.ctm_name,
                args.ablation,
                max_workers=args.max_workers,
                results_dir=args.results_dir,
                resume=args.resume,
                modality=mod,
                threshold_kwargs=threshold_kwargs or None,
            )
    elif args.run_all:
        for ab in ABLATION_CHOICES:
            run_ablation(
                dataset,
                args.dataset_name,
                args.ctm_name,
                [ab],
                max_workers=args.max_workers,
                results_dir=args.results_dir,
                resume=args.resume,
                modality=args.modality,
                threshold_kwargs=threshold_kwargs or None,
            )
    else:
        _run(threshold_kwargs)
