"""Run CTM on a small subset and count LLM API calls via litellm monkey-patch."""
import argparse
import json
import os
import sys
import time

import litellm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CTM modules first so their `from litellm import completion` bindings exist
from dataset_configs import get_dataset_config
from llm_utils import get_audio_path, get_muted_video_path, load_data
import ctm_ai.processors.processor_base as _pb
import ctm_ai.utils.litellm_utils as _llu
import ctm_ai.ctms.ctm_base as _ctmb
from ctm_ai.ctms.ctm import ConsciousTuringMachine


_call_count = 0
_orig_completion = litellm.completion


def _counting_completion(*args, **kwargs):
    global _call_count
    _call_count += 1
    return _orig_completion(*args, **kwargs)


# Patch every module that did `from litellm import completion` at import time
litellm.completion = _counting_completion
_llu.completion = _counting_completion
_pb.completion = _counting_completion
if hasattr(_ctmb, 'completion'):
    _ctmb.completion = _counting_completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='mustard')
    parser.add_argument('--ctm_name', default='sarcasm_ctm')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--output', default='ctm_mustard_gemini_count10.jsonl')
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    dataset = load_data(config.get_default_dataset_path())
    test_ids = list(dataset.keys())[: args.n]

    print(f'Running CTM on {len(test_ids)} {args.dataset_name} samples with {args.ctm_name}')

    per_sample_stats = []
    total_start = time.time()

    for i, tid in enumerate(test_ids, 1):
        global _call_count
        _call_count = 0
        sample = dataset[tid]
        ctm = ConsciousTuringMachine(args.ctm_name)
        target = config.get_text_field(sample)
        query = config.get_task_query()
        audio_path = get_audio_path(tid, args.dataset_name)
        video_path = get_muted_video_path(tid, args.dataset_name)
        if not os.path.exists(audio_path):
            audio_path = None
        if not os.path.exists(video_path):
            video_path = None

        t0 = time.time()
        try:
            answer, weight_score, parsed_answer = ctm(
                query=query,
                text=target,
                video_path=video_path,
                audio_path=audio_path,
                instance_id=tid,
            )
            dt = time.time() - t0
            calls = _call_count
            iters = len(ctm.iteration_history)
            per_sample_stats.append((tid, dt, calls, iters))
            print(f'[{i}/{len(test_ids)}] {tid}: {dt:.1f}s, calls={calls}, iters={iters}, parsed={parsed_answer}')
            with open(args.output, 'a') as f:
                f.write(json.dumps({tid: {
                    'answer': [answer], 'parsed_answer': [parsed_answer],
                    'weight_score': weight_score,
                    'label': config.get_label_field(sample),
                    'num_iterations': iters,
                    'api_calls': calls,
                    'time_sec': dt,
                }}) + '\n')
        except Exception as e:
            print(f'[{i}/{len(test_ids)}] {tid}: FAILED: {e}')

    total_time = time.time() - total_start
    print('\n' + '=' * 60)
    print(f'CTM {args.dataset_name} Gemini — {len(per_sample_stats)} samples')
    print('=' * 60)
    if per_sample_stats:
        total_calls = sum(s[2] for s in per_sample_stats)
        avg_calls = total_calls / len(per_sample_stats)
        avg_time = sum(s[1] for s in per_sample_stats) / len(per_sample_stats)
        avg_iters = sum(s[3] for s in per_sample_stats) / len(per_sample_stats)
        print(f'Total API calls: {total_calls}')
        print(f'Avg API calls/sample: {avg_calls:.2f}')
        print(f'Avg time/sample: {avg_time:.2f}s')
        print(f'Avg iterations: {avg_iters:.2f}')
        print(f'Total wall time: {total_time:.2f}s')


if __name__ == '__main__':
    main()
