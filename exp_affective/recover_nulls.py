"""
Recover null samples in unified_ctmprompt jsonls by re-running them with the
new llm_utils frames fallback.

A null sample in the jsonl is one where ``answer`` is ``[null]`` —
typically because DashScope's video modality pre-check rejected the video
(``The video file is too short``, ``InvalidParameter``, etc.). The updated
``BaseAgent.call`` in ``llm_utils.py`` now automatically falls back to
extracted frames on those errors, so re-running the null sample through the
same pipeline often succeeds.

Usage:
    python recover_nulls.py \\
        --jsonl results/unified_ctmprompt_mustard_qwen3vl_8b_instruct.jsonl \\
        --dataset_name mustard \\
        --model qwen3-vl-8b-instruct

    python recover_nulls.py \\
        --jsonl results/unified_ctmprompt_urfunny_qwen3vl_flash.jsonl \\
        --dataset_name urfunny \\
        --model qwen3-vl-flash
"""

import argparse
import json
import os
import sys
import time

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    check_api_key,
    create_agent,
    load_data,
    load_sample_inputs,
    normalize_label,
)
from run_unified_align_with_ctm_prompt import FramesOnlyMultimodalAgent

sys.path.append('..')


def _load_jsonl(path):
    """Load a jsonl as ordered list of (key, record_dict) pairs preserving
    line order so we can rewrite the file with null entries replaced.
    """
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # each line is {test_file_id: {answer, label, ...}}
            for k, v in d.items():
                entries.append((k, v))
    return entries


def _is_null(record):
    ans = record.get('answer')
    if isinstance(ans, list):
        ans = ans[0] if ans else None
    return ans is None or ans == ''


def _write_jsonl(path, entries):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for k, v in entries:
            f.write(json.dumps({k: v}, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def run(args):
    # Locate CTM_MERGED_PROMPTS by importing the sibling runner module. This
    # keeps the recovery script in lock-step with whatever prompt the original
    # run used, without duplicating 3KB of text.
    from run_unified_align_with_ctm_prompt import CTM_MERGED_PROMPTS

    if args.dataset_name not in CTM_MERGED_PROMPTS:
        raise ValueError(
            f'No CTM_MERGED_PROMPTS entry for dataset {args.dataset_name!r}'
        )
    system_prompt = CTM_MERGED_PROMPTS[args.dataset_name]

    config = get_dataset_config(args.dataset_name)
    dataset_path = args.dataset or config.get_default_dataset_path()
    dataset = load_data(dataset_path)

    entries = _load_jsonl(args.jsonl)
    total = len(entries)
    null_keys = [k for k, v in entries if _is_null(v)]
    print(
        f'{args.jsonl}: {total} entries, {len(null_keys)} null '
        f'({len(null_keys) / total * 100:.1f}%)'
    )
    if not null_keys:
        print('nothing to recover')
        return

    check_api_key(args.provider)
    litellm.set_verbose = False

    # Force frames-only mode for recovery: bypasses DashScope's video-level
    # modality checks entirely (length limits, content moderation on video,
    # connection hiccups during video upload). For samples where the video
    # pipeline fails, sending 8 extracted frames as image_url blocks usually
    # succeeds because the image-level pipeline has different (often more
    # lenient) constraints.
    agent = FramesOnlyMultimodalAgent(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        system_prompt=system_prompt,
    )
    print(f'Model: {agent.model}  |  Dataset: {args.dataset_name}  |  Mode: frames-only')
    print(f'System prompt: CTM-merged ({len(system_prompt)} chars)')
    print('=' * 60)

    # Index for quick replace
    idx_by_key = {k: i for i, (k, _) in enumerate(entries)}

    recovered = 0
    still_null = 0

    for i, key in enumerate(null_keys):
        if key not in dataset:
            print(f'[{i + 1}/{len(null_keys)}] {key}: NOT in dataset, skip')
            still_null += 1
            continue

        inputs = load_sample_inputs(key, dataset, args.dataset_name)
        target_sentence = inputs['target_sentence']
        label = inputs['label']
        full_video_path = inputs['full_video_path']

        query = f"target text: '{target_sentence}'"

        print(
            f'[{i + 1}/{len(null_keys)}] {key}: target={target_sentence[:60]!r}...'
        )

        start = time.time()
        answer, usage = agent.call(query, video_path=full_video_path)
        elapsed = time.time() - start

        if not answer:
            print(f'  STILL NULL after fallback ({elapsed:.1f}s)')
            still_null += 1
        else:
            print(f'  recovered: {answer[:80]}... ({elapsed:.1f}s)')
            recovered += 1

        # Replace the entry in-memory (keep original label/method)
        orig = entries[idx_by_key[key]][1]
        new_record = dict(orig)
        new_record['answer'] = [answer]
        new_record['label_normalized'] = new_record.get(
            'label_normalized'
        ) or normalize_label(label)
        new_record['usage'] = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'api_calls': 1,
        }
        new_record['latency'] = elapsed
        new_record['recovered'] = True
        entries[idx_by_key[key]] = (key, new_record)

        # Persist after each sample so a crash doesn't lose progress
        _write_jsonl(args.jsonl, entries)
        time.sleep(1)

    print()
    print('=' * 60)
    print(f'Recovery summary for {args.jsonl}:')
    print(f'  null samples processed : {len(null_keys)}')
    print(f'  recovered              : {recovered}')
    print(f'  still null             : {still_null}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recover null samples via frames fallback')
    parser.add_argument('--jsonl', type=str, required=True)
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['urfunny', 'mustard'],
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='qwen',
        choices=['gemini', 'qwen'],
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    run(args)
