"""
MoA (Mixture of Agents) baseline for affective detection (MUStARD / URFunny).

1-layer MoA: 3 modality proposers (parallel) + 1 aggregator
Total LLM calls per sample: 4 (3 parallel + 1 sequential)
Backbone: gemini/gemini-2.5-flash-lite via litellm

Proposer system prompts are loaded from the dataset-specific CTM config:
  - mustard  -> sarcasm_ctm_config.json
  - urfunny  -> urfunny_test_qwen_v12_config.json

Examples:
    python run_moa.py --dataset_name mustard --max_workers 8
    python run_moa.py --dataset_name urfunny --max_workers 8
    python run_moa.py --dataset_name mustard --max_workers 4 --resume
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    get_audio_path,
    get_muted_video_path,
    load_data,
)

file_lock = Lock()

MODEL = 'gemini/gemini-2.5-flash-lite'

# =============================================================================
# Load CTM-aligned proposer prompts (same as MetaGPT v2)
# =============================================================================

def _load_ctm_prompts(dataset_name='mustard'):
    config_map = {
        'mustard': 'sarcasm_ctm_config.json',
        'urfunny': 'urfunny_test_qwen_v12_config.json',
    }
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'ctm_conf', config_map.get(dataset_name, config_map['mustard']),
    )
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg['processors_config']


def _get_aggregator_prompt(dataset_name='mustard'):
    if dataset_name == 'urfunny':
        return (
            'You are a humor detection aggregator. You have been provided with independent analyses '
            'from three modality-specific experts (text, audio, video) about whether a punchline is '
            'humorous. Your task is to critically evaluate their analyses and make a final determination.\n\n'
            'Key guidelines:\n'
            '- Weigh evidence from all modalities. Text analysis typically provides the strongest '
            'semantic signal, audio provides tonal/laughter confirmation, video provides supplementary cues.\n'
            '- If experts disagree, consider which modality has stronger evidence for this specific case.\n'
            '- If an expert expresses low confidence or uncertainty, weigh their analysis less.\n'
            '- Do not simply follow the majority — evaluate the quality of each expert\'s reasoning.\n\n'
            'Your answer MUST start with either "Yes" (humorous) or "No" (not humorous), '
            'followed by a brief explanation citing the most decisive evidence.\n\n'
            'Expert analyses:'
        )
    return (
        'You are a sarcasm detection aggregator. You have been provided with independent analyses '
        'from three modality-specific experts (text, audio, video) about whether a person is being '
        'sarcastic. Your task is to critically evaluate their analyses and make a final determination.\n\n'
        'Key guidelines:\n'
        '- Weigh evidence from all modalities. Text analysis typically provides the strongest '
        'semantic signal, audio provides tonal confirmation, video provides supplementary cues.\n'
        '- If experts disagree, consider which modality has stronger evidence for this specific case.\n'
        '- If an expert expresses low confidence or uncertainty, weigh their analysis less.\n'
        '- Do not simply follow the majority — evaluate the quality of each expert\'s reasoning.\n\n'
        'Your answer MUST start with either "Yes" (sarcastic) or "No" (not sarcastic), '
        'followed by a brief explanation citing the most decisive evidence.\n\n'
        'Expert analyses:'
    )


# Defaults (overridden in main based on dataset_name)
_CTM_PROMPTS = _load_ctm_prompts()
TEXT_PROPOSER_SYSTEM = _CTM_PROMPTS['language_processor']['system_prompt']
AUDIO_PROPOSER_SYSTEM = _CTM_PROMPTS['audio_processor']['system_prompt']
VIDEO_PROPOSER_SYSTEM = _CTM_PROMPTS['video_processor']['system_prompt']
MOA_AGGREGATOR_SYSTEM = _get_aggregator_prompt()


# =============================================================================
# LLM Call with Retry
# =============================================================================

def _call_llm(messages, max_retries=3):
    """Call gemini-2.5-flash-lite with retry. Returns (text, usage_dict)."""
    start = time.time()
    api_calls = 0
    for attempt in range(1, max_retries + 1):
        try:
            api_calls += 1
            response = litellm.completion(
                model=MODEL, messages=messages, temperature=0.0,
            )
            text = response.choices[0].message.content
            usage = {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) or 0,
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0) or 0,
                'total_tokens': getattr(response.usage, 'total_tokens', 0) or 0,
                'api_calls': api_calls,
                'latency': time.time() - start,
            }
            if text:
                return text.strip(), usage
        except Exception as e:
            print(f'  LLM attempt {attempt}/{max_retries} failed: {e}')
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None, {
        'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
        'api_calls': api_calls, 'latency': time.time() - start,
    }


# =============================================================================
# Proposers (modality-specific, parallel)
# =============================================================================

def proposer_text(query, target_sentence):
    """Text proposer."""
    messages = [
        {'role': 'system', 'content': TEXT_PROPOSER_SYSTEM},
        {'role': 'user', 'content': (
            f'{query}\n\nThe relevant text of the query is: {target_sentence}'
        )},
    ]
    return _call_llm(messages)


def proposer_audio(query, audio_path):
    """Audio proposer."""
    if not audio_path or not os.path.exists(audio_path):
        empty = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
                 'api_calls': 0, 'latency': 0.0}
        return '[Audio unavailable]', empty

    ext = audio_path.split('.')[-1].lower()
    mime_map = {'mp3': 'audio/mp3', 'wav': 'audio/wav', 'mp4': 'audio/mp4', 'aac': 'audio/aac'}
    mime_type = mime_map.get(ext, 'audio/mp4')
    with open(audio_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    messages = [
        {'role': 'system', 'content': AUDIO_PROPOSER_SYSTEM},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': f'{query}\n\nBased on the audio, provide your analysis.'},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{encoded}'}},
        ]},
    ]
    return _call_llm(messages)


def proposer_video(query, video_path):
    """Video proposer."""
    if not video_path or not os.path.exists(video_path):
        empty = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
                 'api_calls': 0, 'latency': 0.0}
        return '[Video unavailable]', empty

    with open(video_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    messages = [
        {'role': 'system', 'content': VIDEO_PROPOSER_SYSTEM},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': f'{query}\n\nAnalyze the muted video.'},
            {'type': 'image_url', 'image_url': {'url': f'data:video/mp4;base64,{encoded}'}},
        ]},
    ]
    return _call_llm(messages)


# =============================================================================
# Aggregator
# =============================================================================

def aggregator(query, proposer_outputs):
    """MoA aggregator: synthesize all proposer outputs into final answer."""
    # Build system prompt with numbered references (faithful to MoA paper)
    system = MOA_AGGREGATOR_SYSTEM
    for i, (modality, output) in enumerate(proposer_outputs):
        system += f'\n{i+1}. [{modality}] {output}'

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': query},
    ]
    return _call_llm(messages)


# =============================================================================
# Full MoA Pipeline
# =============================================================================

def run_moa_pipeline(query, target_sentence, audio_path, video_path):
    """Run 1-layer MoA: 3 proposers in parallel, then aggregator.

    Returns:
        tuple: (final_answer, stage_outputs, usage_stats)
    """
    all_usages = []
    round_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(proposer_text, query, target_sentence): 'text',
            executor.submit(proposer_audio, query, audio_path): 'audio',
            executor.submit(proposer_video, query, video_path): 'video',
        }
        for future in as_completed(futures):
            modality = futures[future]
            text, usage = future.result()
            round_results[modality] = text or f'[{modality} failed]'
            all_usages.append((modality, usage))

    # Final aggregation
    proposer_outputs = [
        ('Text Analysis', round_results['text']),
        ('Audio Analysis', round_results['audio']),
        ('Video Analysis', round_results['video']),
    ]
    final_answer, agg_usage = aggregator(query, proposer_outputs)
    all_usages.append(('aggregator', agg_usage))

    usage_stats = {
        'total_api_calls': sum(u.get('api_calls', 0) for _, u in all_usages),
        'total_prompt_tokens': sum(u.get('prompt_tokens', 0) for _, u in all_usages),
        'total_completion_tokens': sum(u.get('completion_tokens', 0) for _, u in all_usages),
        'total_tokens': sum(u.get('total_tokens', 0) for _, u in all_usages),
        'stage_latencies': {name: u.get('latency', 0.0) for name, u in all_usages},
    }

    stage_outputs = {
        'text_proposer': round_results['text'],
        'audio_proposer': round_results['audio'],
        'video_proposer': round_results['video'],
    }

    return final_answer, stage_outputs, usage_stats


# =============================================================================
# Instance Runner
# =============================================================================

def run_instance(test_file, dataset, dataset_name, output_file):
    try:
        config = get_dataset_config(dataset_name)
        sample = dataset[test_file]
        target_sentence = config.get_text_field(sample)
        label = config.get_label_field(sample)
        query = config.get_task_query()

        audio_path = get_audio_path(test_file, dataset_name)
        video_path = get_muted_video_path(test_file, dataset_name)
        if not os.path.exists(audio_path):
            audio_path = None
        if not os.path.exists(video_path):
            video_path = None

        start_time = time.time()
        final_answer, stage_outputs, usage_stats = run_moa_pipeline(
            query, target_sentence, audio_path, video_path,
        )
        elapsed = time.time() - start_time

        parsed = final_answer.strip() if final_answer else ''
        print(
            f'[{test_file}] {elapsed:.1f}s | '
            f'api={usage_stats["total_api_calls"]} '
            f'tok={usage_stats["total_tokens"]} | '
            f'{parsed[:60]}'
        )

        result = {
            test_file: {
                'answer': [final_answer or ''],
                'parsed_answer': [parsed],
                'label': label,
                'method': 'moa_1round',
                'latency': elapsed,
                'api_calls': usage_stats['total_api_calls'],
                'prompt_tokens': usage_stats['total_prompt_tokens'],
                'completion_tokens': usage_stats['total_completion_tokens'],
                'total_tokens': usage_stats['total_tokens'],
                'stage_latencies': usage_stats['stage_latencies'],
                'stage_outputs': stage_outputs,
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


# =============================================================================
# Parallel Runner
# =============================================================================

def load_processed_keys(output_file):
    processed = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    processed.update(json.loads(line).keys())
    except FileNotFoundError:
        pass
    return processed


def run_parallel(dataset, dataset_name, max_workers=4,
                 output_file='moa_mustard.jsonl', resume=False,
                 sleep_between=0):
    test_list = list(dataset.keys())

    if resume:
        processed = load_processed_keys(output_file)
        test_list = [k for k in test_list if k not in processed]
        print(f'Resume: {len(processed)} done, {len(test_list)} remaining')
    else:
        with open(output_file, 'w'):
            pass

    if not test_list:
        print('Nothing to do.')
        return

    print('=' * 60)
    print(f'MoA 1-round | Model: {MODEL}')
    print(f'Samples: {len(test_list)} | Workers: {max_workers}')
    print(f'Output: {output_file}')
    print('=' * 60)

    start_time = time.time()
    completed = 0

    if max_workers == 1 and sleep_between > 0:
        for test_file in test_list:
            try:
                result = run_instance(test_file, dataset, dataset_name, output_file)
                completed += 1
                print(f'Progress: {completed}/{len(test_list)} - {result}')
            except Exception as exc:
                completed += 1
                print(f'Error: {exc}')
            if sleep_between > 0 and completed < len(test_list):
                time.sleep(sleep_between)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_instance, t, dataset, dataset_name, output_file): t
                for t in test_list
            }
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    print(f'Progress: {completed}/{len(test_list)} - {result}')
                except Exception as exc:
                    print(f'Error: {exc}')

    wall_time = time.time() - start_time
    _print_summary(output_file, wall_time)


def _print_summary(output_file, wall_time):
    total_api = total_pt = total_ct = total_tok = 0
    latencies = []
    stage_lats = {}
    n = 0
    try:
        with open(output_file) as f:
            for line in f:
                if not line.strip():
                    continue
                for _, val in json.loads(line).items():
                    n += 1
                    total_api += val.get('api_calls', 0)
                    total_pt += val.get('prompt_tokens', 0)
                    total_ct += val.get('completion_tokens', 0)
                    total_tok += val.get('total_tokens', 0)
                    latencies.append(val.get('latency', 0.0))
                    for s, l in val.get('stage_latencies', {}).items():
                        stage_lats.setdefault(s, []).append(l)
    except Exception as e:
        print(f'Warning: {e}')
        return

    if n == 0:
        return

    cost = total_pt / 1e6 * 0.075 + total_ct / 1e6 * 0.30
    print(f'\n{"=" * 60}')
    print(f'MOA PERFORMANCE & COST SUMMARY')
    print(f'{"=" * 60}')
    print(f'Samples:                  {n}')
    print(f'Wall-clock:               {wall_time:.1f}s')
    print(f'-' * 40)
    print(f'Total API Calls:          {total_api}')
    print(f'Avg API Calls/Sample:     {total_api / n:.1f}')
    print(f'Avg Latency/Sample:       {sum(latencies) / n:.2f}s')
    print(f'-' * 40)
    print(f'Total Prompt Tokens:      {total_pt:,}')
    print(f'Total Completion Tokens:  {total_ct:,}')
    print(f'Total Tokens:             {total_tok:,}')
    print(f'Avg Tokens/Sample:        {total_tok / n:,.0f}')
    print(f'-' * 40)
    print(f'Estimated Cost:           ${cost:.4f}')
    print(f'Avg Cost/Sample:          ${cost / n:.6f}')
    if stage_lats:
        print(f'-' * 40)
        print('Avg Latency by Stage:')
        for s in sorted(stage_lats):
            vals = stage_lats[s]
            print(f'  {s:30s} {sum(vals)/len(vals):.2f}s')
    print(f'{"=" * 60}\n')


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoA 1-round baseline for affective detection')
    parser.add_argument('--dataset_name', type=str, default='mustard', choices=['urfunny', 'mustard'])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sleep', type=float, default=0)
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    # Load prompts for the target dataset
    import sys as _sys
    _this = _sys.modules[__name__]
    _prompts = _load_ctm_prompts(args.dataset_name)
    _this.TEXT_PROPOSER_SYSTEM = _prompts['language_processor']['system_prompt']
    _this.AUDIO_PROPOSER_SYSTEM = _prompts['audio_processor']['system_prompt']
    _this.VIDEO_PROPOSER_SYSTEM = _prompts['video_processor']['system_prompt']
    _this.MOA_AGGREGATOR_SYSTEM = _get_aggregator_prompt(args.dataset_name)

    output_file = args.output or f'moa_{args.dataset_name}_1round.jsonl'

    dataset = load_data(args.dataset)
    print(f'Dataset: {args.dataset_name} | Model: {MODEL} | Samples: {len(dataset)}')

    run_parallel(
        dataset, args.dataset_name,
        max_workers=args.max_workers, output_file=output_file,
        resume=args.resume, sleep_between=args.sleep,
    )
