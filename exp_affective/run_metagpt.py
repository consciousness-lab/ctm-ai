"""
MetaGPT-style multi-agent baseline for sarcasm detection on MUStARD.

4-stage SOP pipeline:
  Stage 1: Independent modality analysis (3 agents in parallel)
  Stage 2: Cross-modal synthesis (1 agent)
  Stage 3: Critical review (1 agent)
  Stage 4: Final judgment (1 agent)

Total LLM calls per sample: 6 (3 parallel + 3 sequential)
Backbone: gemini/gemini-2.5-flash-lite via litellm

Examples:
    python run_metagpt.py --dataset_name mustard --max_workers 8
    python run_metagpt.py --dataset_name mustard --max_workers 4 --resume
    python run_metagpt.py --dataset_name mustard --max_workers 1 --sleep 2
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
# MetaGPT Agent Prompts (Role-based, structured communication)
# =============================================================================

# Stage 1 prompts: use the SAME system prompts as CTM processors (exp22) for fair comparison.
# Only the architecture (SOP pipeline vs competition/fusion) should differ.
def _load_ctm_prompts(dataset_name='mustard'):
    """Load system prompts from CTM config to stay aligned."""
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


def _get_metagpt_prompts(dataset_name='mustard'):
    if dataset_name == 'urfunny':
        task = 'humor detection'
        yes_label = 'humorous'
        no_label = 'not humorous'
        pitfalls = (
            '   - Enthusiasm/excitement with humor\n'
            '   - Deadpan delivery with comedic intent\n'
            '   - Informative surprises with actual jokes\n'
            '   - Rhetorical contrasts with humorous incongruity'
        )
    else:
        task = 'sarcasm detection'
        yes_label = 'sarcastic'
        no_label = 'not sarcastic'
        pitfalls = (
            '   - Genuine anger/frustration with sarcasm\n'
            '   - Dry/deadpan delivery with sarcastic intent\n'
            '   - Playful teasing with sarcasm\n'
            '   - Short responses as automatically non-sarcastic'
        )

    synthesizer = (
        f'You are the Synthesizer in a multi-agent {task} team. '
        'You have received independent analysis reports from three modality experts '
        '(Text, Audio, Video). Your role is to integrate their findings into a '
        'coherent cross-modal synthesis.\n\n'
        'Your synthesis MUST cover:\n'
        '1. **Agreement Points**: Where do the modality experts agree? What evidence is consistent?\n'
        '2. **Conflicts**: Where do experts disagree? Which modality provides stronger evidence?\n'
        f'3. **Evidence Strength**: Rate the overall evidence for {yes_label} vs {no_label}.\n'
        '4. **Modality Reliability**: For this specific sample, which modality seems most reliable and why?\n'
        f'5. **Synthesis Verdict**: Based on integrated evidence, is the person likely {yes_label} or not?\n\n'
        'Weigh the evidence carefully. Text analysis typically provides the strongest semantic signal, '
        'audio provides tonal confirmation/contradiction, and video provides supplementary visual cues.'
    )
    critic = (
        f'You are the Critic in a multi-agent {task} team. '
        'You have received the original modality analyses AND the synthesizer\'s integration. '
        'Your role is to critically review the synthesis and challenge weak reasoning.\n\n'
        'Your review MUST cover:\n'
        '1. **Logical Consistency**: Is the synthesis logically sound? Are conclusions supported by evidence?\n'
        '2. **Overlooked Evidence**: Did the synthesizer miss or underweight important evidence from any modality?\n'
        '3. **Common Pitfalls**: Flag if the analysis might be confusing:\n'
        f'{pitfalls}\n'
        '4. **Uncertainty Assessment**: How uncertain is the overall judgment? Are there genuine ambiguities?\n'
        '5. **Revised Verdict**: After your critical review, do you agree with the synthesis or disagree? Why?\n\n'
        'Be rigorous. Your job is to catch errors before the final decision is made.'
    )
    judge = (
        f'You are the Final Judge in a multi-agent {task} team. '
        'You have received all prior analyses: three modality reports, a cross-modal synthesis, '
        f'and a critical review. Your role is to make the final {task} determination.\n\n'
        'Review all evidence carefully and make your decision.\n\n'
        f'Your answer MUST start with either "Yes" ({yes_label}) or "No" ({no_label}), '
        'followed by a brief explanation citing the most decisive evidence.\n\n'
        'Guidelines:\n'
        f'- If the Critic identified significant weaknesses in the {yes_label} case, weigh that heavily.\n'
        '- If modalities strongly agree, trust the consensus.\n'
        '- If modalities conflict, prioritize text > audio > video for semantic judgment.\n'
        '- Make a definitive decision — do not hedge.'
    )
    return synthesizer, critic, judge


_CTM_PROMPTS = _load_ctm_prompts()
TEXT_ANALYST_SYSTEM = _CTM_PROMPTS['language_processor']['system_prompt']
AUDIO_ANALYST_SYSTEM = _CTM_PROMPTS['audio_processor']['system_prompt']
VIDEO_ANALYST_SYSTEM = _CTM_PROMPTS['video_processor']['system_prompt']
SYNTHESIZER_SYSTEM, CRITIC_SYSTEM, JUDGE_SYSTEM = _get_metagpt_prompts()


# =============================================================================
# LLM Call with Retry (returns text + usage stats)
# =============================================================================

def _call_llm(messages, max_retries=3):
    """Call gemini-2.5-flash-lite with retry.

    Returns:
        tuple: (text, usage_dict) where usage_dict has
               prompt_tokens, completion_tokens, total_tokens, api_calls, latency
    """
    start = time.time()
    api_calls = 0
    for attempt in range(1, max_retries + 1):
        try:
            api_calls += 1
            response = litellm.completion(
                model=MODEL,
                messages=messages,
                temperature=0.0,
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
# Stage 1: Independent Modality Analysis (parallel)
# =============================================================================

def stage1_text_analysis(query, target_sentence):
    """Text Analyst: analyze dialogue transcript."""
    messages = [
        {'role': 'system', 'content': TEXT_ANALYST_SYSTEM},
        {'role': 'user', 'content': (
            f'{query}\n\n'
            f'Dialogue transcript:\n{target_sentence}'
        )},
    ]
    text, usage = _call_llm(messages)
    return text, usage


def stage1_audio_analysis(query, audio_path):
    """Audio Analyst: analyze audio recording."""
    if not audio_path or not os.path.exists(audio_path):
        empty_usage = {
            'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
            'api_calls': 0, 'latency': 0.0,
        }
        return '[Audio unavailable — no audio file found for this sample.]', empty_usage

    ext = audio_path.split('.')[-1].lower()
    mime_map = {
        'mp3': 'audio/mp3', 'wav': 'audio/wav',
        'mp4': 'audio/mp4', 'aac': 'audio/aac',
    }
    mime_type = mime_map.get(ext, 'audio/mp4')
    with open(audio_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    messages = [
        {'role': 'system', 'content': AUDIO_ANALYST_SYSTEM},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': (
                f'{query}\n\nAnalyze the audio recording and produce your structured report.'
            )},
            {'type': 'image_url', 'image_url': {
                'url': f'data:{mime_type};base64,{encoded}',
            }},
        ]},
    ]
    text, usage = _call_llm(messages)
    return text, usage


def stage1_video_analysis(query, video_path):
    """Video Analyst: analyze muted video."""
    if not video_path or not os.path.exists(video_path):
        empty_usage = {
            'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
            'api_calls': 0, 'latency': 0.0,
        }
        return '[Video unavailable — no video file found for this sample.]', empty_usage

    with open(video_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    messages = [
        {'role': 'system', 'content': VIDEO_ANALYST_SYSTEM},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': (
                f'{query}\n\nAnalyze the muted video and produce your structured report.'
            )},
            {'type': 'image_url', 'image_url': {
                'url': f'data:video/mp4;base64,{encoded}',
            }},
        ]},
    ]
    text, usage = _call_llm(messages)
    return text, usage


# =============================================================================
# Stage 2: Cross-Modal Synthesis
# =============================================================================

def stage2_synthesis(query, text_report, audio_report, video_report):
    """Synthesizer: integrate all modality reports."""
    messages = [
        {'role': 'system', 'content': SYNTHESIZER_SYSTEM},
        {'role': 'user', 'content': (
            f'Task: {query}\n\n'
            f'=== TEXT ANALYST REPORT ===\n{text_report}\n\n'
            f'=== AUDIO ANALYST REPORT ===\n{audio_report}\n\n'
            f'=== VIDEO ANALYST REPORT ===\n{video_report}\n\n'
            'Produce your cross-modal synthesis.'
        )},
    ]
    text, usage = _call_llm(messages)
    return text, usage


# =============================================================================
# Stage 3: Critical Review
# =============================================================================

def stage3_critic(query, text_report, audio_report, video_report, synthesis):
    """Critic: review and challenge the synthesis."""
    messages = [
        {'role': 'system', 'content': CRITIC_SYSTEM},
        {'role': 'user', 'content': (
            f'Task: {query}\n\n'
            f'=== TEXT ANALYST REPORT ===\n{text_report}\n\n'
            f'=== AUDIO ANALYST REPORT ===\n{audio_report}\n\n'
            f'=== VIDEO ANALYST REPORT ===\n{video_report}\n\n'
            f'=== SYNTHESIZER REPORT ===\n{synthesis}\n\n'
            'Critically review the synthesis. Identify weaknesses, overlooked evidence, '
            'and potential errors.'
        )},
    ]
    text, usage = _call_llm(messages)
    return text, usage


# =============================================================================
# Stage 4: Final Judgment
# =============================================================================

def stage4_judge(query, text_report, audio_report, video_report, synthesis, critique):
    """Judge: make final Yes/No decision."""
    messages = [
        {'role': 'system', 'content': JUDGE_SYSTEM},
        {'role': 'user', 'content': (
            f'Task: {query}\n\n'
            f'=== TEXT ANALYST REPORT ===\n{text_report}\n\n'
            f'=== AUDIO ANALYST REPORT ===\n{audio_report}\n\n'
            f'=== VIDEO ANALYST REPORT ===\n{video_report}\n\n'
            f'=== SYNTHESIZER REPORT ===\n{synthesis}\n\n'
            f'=== CRITIC REVIEW ===\n{critique}\n\n'
            'Make your final determination. Start with "Yes" or "No".'
        )},
    ]
    text, usage = _call_llm(messages)
    return text, usage


# =============================================================================
# Full MetaGPT Pipeline
# =============================================================================

def _merge_usage(all_usages):
    """Aggregate usage stats from all stages."""
    total = {
        'total_api_calls': 0,
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'stage_latencies': {},
    }
    for stage_name, usage in all_usages:
        total['total_api_calls'] += usage.get('api_calls', 0)
        total['total_prompt_tokens'] += usage.get('prompt_tokens', 0)
        total['total_completion_tokens'] += usage.get('completion_tokens', 0)
        total['total_tokens'] += usage.get('total_tokens', 0)
        total['stage_latencies'][stage_name] = usage.get('latency', 0.0)
    return total


def run_metagpt_pipeline(query, target_sentence, audio_path, video_path):
    """Run the full 4-stage MetaGPT SOP pipeline for one sample.

    Returns:
        tuple: (final_answer, stage_outputs_dict, usage_stats_dict)
    """
    all_usages = []

    # Stage 1: Independent analysis (parallel via threads)
    stage1_results = {}
    stage1_usages = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(stage1_text_analysis, query, target_sentence): 'text',
            executor.submit(stage1_audio_analysis, query, audio_path): 'audio',
            executor.submit(stage1_video_analysis, query, video_path): 'video',
        }
        for future in as_completed(futures):
            modality = futures[future]
            text, usage = future.result()
            stage1_results[modality] = text or f'[{modality} analysis failed]'
            stage1_usages[modality] = usage
            all_usages.append((f'stage1_{modality}', usage))

    text_report = stage1_results['text']
    audio_report = stage1_results['audio']
    video_report = stage1_results['video']

    # Stage 2: Cross-modal synthesis
    synthesis, s2_usage = stage2_synthesis(query, text_report, audio_report, video_report)
    all_usages.append(('stage2_synthesis', s2_usage))
    if not synthesis:
        synthesis = '[Synthesis failed]'

    # Stage 3: Critical review
    critique, s3_usage = stage3_critic(
        query, text_report, audio_report, video_report, synthesis,
    )
    all_usages.append(('stage3_critic', s3_usage))
    if not critique:
        critique = '[Critique failed]'

    # Stage 4: Final judgment
    final_answer, s4_usage = stage4_judge(
        query, text_report, audio_report, video_report, synthesis, critique,
    )
    all_usages.append(('stage4_judge', s4_usage))

    stage_outputs = {
        'text_report': text_report,
        'audio_report': audio_report,
        'video_report': video_report,
        'synthesis': synthesis,
        'critique': critique,
    }

    usage_stats = _merge_usage(all_usages)

    return final_answer, stage_outputs, usage_stats


# =============================================================================
# Instance Runner
# =============================================================================

def run_instance(test_file, dataset, dataset_name, output_file):
    """Process a single sample through the MetaGPT pipeline."""
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

        final_answer, stage_outputs, usage_stats = run_metagpt_pipeline(
            query, target_sentence, audio_path, video_path,
        )

        elapsed = time.time() - start_time

        parsed = final_answer.strip() if final_answer else ''

        print(
            f'[{test_file}] {elapsed:.1f}s | '
            f'api_calls={usage_stats["total_api_calls"]} '
            f'tokens={usage_stats["total_tokens"]} | '
            f'{parsed[:60]}'
        )

        result = {
            test_file: {
                'answer': [final_answer or ''],
                'parsed_answer': [parsed],
                'label': label,
                'method': 'metagpt',
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
    """Load already processed keys from JSONL output file for resume."""
    processed = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    processed.update(result.keys())
    except FileNotFoundError:
        pass
    return processed


def run_parallel(
    dataset, dataset_name, max_workers=4,
    output_file='metagpt_mustard.jsonl', resume=False, sleep_between=0,
):
    test_list = list(dataset.keys())
    print(f'Total test samples: {len(test_list)}')
    print(f'Using {max_workers} workers')
    print(f'Output file: {output_file}')

    if resume:
        processed = load_processed_keys(output_file)
        test_list = [k for k in test_list if k not in processed]
        print(f'Resume mode: {len(processed)} already done, {len(test_list)} remaining')
    else:
        with open(output_file, 'w', encoding='utf-8'):
            pass

    if not test_list:
        print('Nothing to do.')
        return

    print('=' * 60)
    print(f'MetaGPT 4-stage SOP pipeline | Model: {MODEL}')
    print(f'Samples: {len(test_list)} | Workers: {max_workers}')
    print('=' * 60)

    start_time = time.time()
    completed_count = 0

    if max_workers == 1 and sleep_between > 0:
        for test_file in test_list:
            try:
                result = run_instance(
                    test_file, dataset, dataset_name, output_file,
                )
                completed_count += 1
                print(f'Progress: {completed_count}/{len(test_list)} - {result}')
            except Exception as exc:
                completed_count += 1
                print(f'Error processing {test_file}: {exc}')
            if sleep_between > 0 and completed_count < len(test_list):
                time.sleep(sleep_between)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {
                executor.submit(
                    run_instance, test_file, dataset, dataset_name, output_file,
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

    total_time = time.time() - start_time

    # Print aggregate stats from the output file
    _print_aggregate_stats(output_file, total_time)


def _print_aggregate_stats(output_file, wall_time):
    """Read the JSONL output and print aggregate API call / latency / token stats."""
    total_api_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    latencies = []
    stage_latency_sums = {}
    sample_count = 0

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for _key, val in record.items():
                    sample_count += 1
                    total_api_calls += val.get('api_calls', 0)
                    total_prompt_tokens += val.get('prompt_tokens', 0)
                    total_completion_tokens += val.get('completion_tokens', 0)
                    total_tokens += val.get('total_tokens', 0)
                    latencies.append(val.get('latency', 0.0))
                    for stage, lat in val.get('stage_latencies', {}).items():
                        stage_latency_sums.setdefault(stage, []).append(lat)
    except Exception as e:
        print(f'Warning: could not read stats from {output_file}: {e}')
        return

    if sample_count == 0:
        print('No results to summarize.')
        return

    avg_latency = sum(latencies) / sample_count
    avg_api_calls = total_api_calls / sample_count

    # Gemini 2.5 Flash Lite pricing (per 1M tokens)
    cost_input_per_1m = 0.075
    cost_output_per_1m = 0.30
    total_cost = (
        total_prompt_tokens / 1_000_000 * cost_input_per_1m
        + total_completion_tokens / 1_000_000 * cost_output_per_1m
    )

    print('\n' + '=' * 60)
    print('METAGPT PERFORMANCE & COST SUMMARY')
    print('=' * 60)
    print(f'Total Samples:            {sample_count}')
    print(f'Wall-clock Time:          {wall_time:.1f}s')
    print('-' * 40)
    print(f'Total API Calls:          {total_api_calls}')
    print(f'Avg API Calls/Sample:     {avg_api_calls:.1f}')
    print(f'Avg Latency/Sample:       {avg_latency:.2f}s')
    print('-' * 40)
    print(f'Total Prompt Tokens:      {total_prompt_tokens:,}')
    print(f'Total Completion Tokens:  {total_completion_tokens:,}')
    print(f'Total Tokens:             {total_tokens:,}')
    print(f'Avg Tokens/Sample:        {total_tokens / sample_count:,.0f}')
    print('-' * 40)
    print(f'Estimated Cost:           ${total_cost:.4f}')
    print(f'Avg Cost/Sample:          ${total_cost / sample_count:.6f}')

    if stage_latency_sums:
        print('-' * 40)
        print('Avg Latency by Stage:')
        for stage in sorted(stage_latency_sums.keys()):
            vals = stage_latency_sums[stage]
            print(f'  {stage:25s} {sum(vals) / len(vals):.2f}s')

    print('=' * 60 + '\n')


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MetaGPT-style multi-agent baseline for sarcasm detection',
    )
    parser.add_argument(
        '--dataset_name', type=str, default='mustard',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: mustard)',
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Path to dataset JSON file (default: auto)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--max_workers', type=int, default=8,
        help='Number of parallel workers for samples (default: 8)',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from existing output file',
    )
    parser.add_argument(
        '--sleep', type=float, default=0,
        help='Sleep seconds between samples (for rate limiting)',
    )
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    # Reload prompts for the target dataset
    import sys as _sys
    _this = _sys.modules[__name__]
    _prompts = _load_ctm_prompts(args.dataset_name)
    _this.TEXT_ANALYST_SYSTEM = _prompts['language_processor']['system_prompt']
    _this.AUDIO_ANALYST_SYSTEM = _prompts['audio_processor']['system_prompt']
    _this.VIDEO_ANALYST_SYSTEM = _prompts['video_processor']['system_prompt']
    _this.SYNTHESIZER_SYSTEM, _this.CRITIC_SYSTEM, _this.JUDGE_SYSTEM = _get_metagpt_prompts(args.dataset_name)

    output_file = args.output or f'metagpt_{args.dataset_name}.jsonl'

    dataset = load_data(args.dataset)
    print(f'Dataset: {args.dataset_name} | Model: {MODEL} | Samples: {len(dataset)}')

    run_parallel(
        dataset,
        args.dataset_name,
        max_workers=args.max_workers,
        output_file=output_file,
        resume=args.resume,
        sleep_between=args.sleep,
    )
