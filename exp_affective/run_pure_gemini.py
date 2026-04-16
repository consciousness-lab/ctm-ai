"""
Pure-Gemini single-call baseline for affective detection (MUStARD / URFunny).

This is the "no orchestration" baseline: one Gemini call per sample that
receives all three modalities (text-with-context, audio, muted video) at
the same time and produces a final Yes/No sarcasm (or humor) verdict.

Prompts are built entirely from CTM's sarcasm_ctm_config.json
(and urfunny_test_gemini_v28_config.json for --dataset_name urfunny):

  * System prompt  = `language_processor.system_prompt`
                   + `audio_processor.system_prompt`
                   + `video_processor.system_prompt`
                   + the output-format block extracted from CTM's
                     `parse_prompt_template` (the "must start with Yes/No,
                     default to No if uncertain" section).

  * User message   = multimodal content:
                       - text part:   task query + dialogue with context
                       - audio part:  base64-encoded .mp4 audio
                       - video part:  base64-encoded .mp4 muted video

Every token in the system prompt comes verbatim from CTM config, so this
baseline shares its modality prompts 1-1 with the MoA, MetaGPT, and
AutoGen baselines' CTM-aligned variants. The difference is that Pure
Gemini does NO orchestration — no proposers, no critic, no debate, no
synthesis. One call in, one Yes/No out.

Examples:
    python run_pure_gemini.py --dataset_name mustard --max_workers 16
    python run_pure_gemini.py --dataset_name urfunny --max_workers 8
    python run_pure_gemini.py --dataset_name mustard --resume
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

import litellm  # noqa: E402
from dataset_configs import get_dataset_config  # noqa: E402
from llm_utils import (  # noqa: E402
    get_audio_path,
    get_muted_video_path,
    load_data,
)

file_lock = Lock()

DEFAULT_MODEL = 'gemini/gemini-2.5-flash-lite'


# =============================================================================
# Prompts — built entirely from CTM config
# =============================================================================

def _load_ctm_config(dataset_name='mustard'):
    config_map = {
        'mustard': 'sarcasm_ctm_config.json',
        # urfunny: use the Gemini-tuned config (was wrongly using
        # urfunny_test_qwen_v12_config.json which is calibrated for Qwen).
        'urfunny': 'urfunny_test_gemini_v28_config.json',
    }
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'ctm_conf',
        config_map.get(dataset_name, config_map['mustard']),
    )
    with open(config_path) as f:
        return json.load(f)


def _extract_output_format_block(parse_prompt_template: str) -> str:
    """Strip off the trailing 'Analysis:\n{answer}' from CTM's
    parse_prompt_template, keeping only the preamble that defines:
      - the expert role
      - the output-format rules ("must start with Yes/No", "default to No")

    This way the Pure Gemini baseline reuses CTM's exact wording for
    HOW TO ANSWER without trying to feed it a pre-computed analysis
    (which we don't have — the raw modalities are the "analysis")."""
    # Split on the 'Analysis:' marker which precedes the {answer} placeholder.
    head = parse_prompt_template.split('Analysis:')[0].rstrip()
    return head


def _build_system_prompt(dataset_name='mustard') -> str:
    """Concatenate CTM's 3 modality processor prompts and the output-format
    rules from parse_prompt_template into a single system prompt for the
    all-in-one Gemini call.

    Format requirement is given BOTH at the top (so it's the first thing
    the model sees) and at the bottom (so it's the last thing in context
    before generation). Empirically, putting it only at the bottom of a
    long multimodal prompt makes Gemini-2.5-flash-lite ignore it ~60% of
    the time and dump raw analysis with no Yes/No marker at all."""
    cfg = _load_ctm_config(dataset_name)
    procs = cfg['processors_config']
    output_rules = _extract_output_format_block(cfg['parse_prompt_template'])

    task_label = 'humor detection' if dataset_name == 'urfunny' else 'sarcasm detection'
    yes_label = 'humorous' if dataset_name == 'urfunny' else 'sarcastic'
    no_label = 'not humorous' if dataset_name == 'urfunny' else 'not sarcastic'

    strict_format = (
        f'OUTPUT FORMAT (STRICTLY ENFORCED): Your response MUST consist of '
        f'exactly two lines and nothing else.\n'
        f'Line 1 must be exactly: "VERDICT: Yes"  ({yes_label})  OR  '
        f'"VERDICT: No"  ({no_label})\n'
        f'Line 2 must be a one-sentence justification citing the most '
        f'decisive evidence across the three modalities.\n'
        f'Do NOT include any analysis, preamble, headings, or markdown '
        f'before Line 1. The very first six characters of your response '
        f'MUST be "VERDICT".'
    )

    return (
        f'{strict_format}\n\n'
        f'---\n\n'
        f'You are a {task_label} system that receives three modalities at '
        f'the same time: dialogue text (with prior context), audio, and a '
        f'muted video. Use each modality according to the specialised '
        f'instructions below, then produce a single final answer.\n\n'
        f'=== LANGUAGE / TEXT ANALYSIS (use for the dialogue transcript) ===\n'
        f'{procs["language_processor"]["system_prompt"]}\n\n'
        f'=== AUDIO ANALYSIS (use for the audio attachment) ===\n'
        f'{procs["audio_processor"]["system_prompt"]}\n\n'
        f'=== VISUAL ANALYSIS (use for the muted video attachment) ===\n'
        f'{procs["video_processor"]["system_prompt"]}\n\n'
        f'=== FINAL DECISION (how to answer) ===\n'
        f'{output_rules}\n\n'
        f'---\n\n'
        f'{strict_format}'
    )


# =============================================================================
# Multimodal message builder
# =============================================================================

def _audio_mime(path: str) -> str:
    ext = path.rsplit('.', 1)[-1].lower()
    return {
        'mp3': 'audio/mp3', 'wav': 'audio/wav',
        'mp4': 'audio/mp4', 'aac': 'audio/aac',
    }.get(ext, 'audio/mp4')


def _build_messages(
    system_prompt: str,
    query: str,
    target_text: str,
    audio_path,
    video_path,
):
    """Build the [system, user] messages with multimodal user content.

    `target_text` is the bare target utterance (mustard) or punchline
    (urfunny) WITHOUT surrounding conversational context — this matches
    what MoA, MetaGPT, and AutoGen feed their text experts via
    `dataset_configs.*.get_text_field()`, so all four baselines see the
    same text input."""
    text_part = (
        f'{query}\n\n'
        f'Target utterance:\n{target_text}'
    )

    content = [{'type': 'text', 'text': text_part}]

    if audio_path and os.path.exists(audio_path):
        with open(audio_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:{_audio_mime(audio_path)};base64,{encoded}'},
        })

    if video_path and os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:video/mp4;base64,{encoded}'},
        })

    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': content},
    ]


# =============================================================================
# LLM call
# =============================================================================

def _call_llm(messages, model, max_retries=3):
    start = time.time()
    api_calls = 0
    for attempt in range(1, max_retries + 1):
        try:
            api_calls += 1
            resp = litellm.completion(
                model=model, messages=messages, temperature=0.2,
                timeout=90,
            )
            text = resp.choices[0].message.content
            usage = {
                'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0) or 0,
                'completion_tokens': getattr(resp.usage, 'completion_tokens', 0) or 0,
                'total_tokens': getattr(resp.usage, 'total_tokens', 0) or 0,
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
# Verdict extraction
# =============================================================================

import re

_VERDICT_LINE_RE = re.compile(r'verdict\s*[:\-]\s*(yes|no)\b', re.IGNORECASE)


def _extract_verdict(text: str) -> str:
    """Extract a Yes/No verdict from the model response.

    Tries (in order):
        1. The strict 'VERDICT: Yes/No' format we asked for in the prompt
        2. A bare 'Yes'/'No' at the start of the response
        3. A 'Final answer: Yes/No' style line anywhere in the text
        4. Phrase-based inference ('is sarcastic', 'is not sarcastic', etc.)

    Returns 'Yes', 'No', or '' if nothing matches.

    The returned string is what `parsed_answer[0]` will hold, so
    calc_ctm_res.extract_prediction (which only checks the first chars
    for 'yes'/'no') can read it directly."""
    if not text:
        return ''
    t = text.strip()

    # 1. VERDICT: Yes / VERDICT: No (the format we requested)
    m = _VERDICT_LINE_RE.search(t)
    if m:
        return 'Yes' if m.group(1).lower() == 'yes' else 'No'

    # 2. Bare Yes/No at the start
    head = t[:8].lower()
    if head.startswith('yes'):
        return 'Yes'
    if head.startswith('no'):
        return 'No'

    # 3. "Final answer: Yes" / "My answer: No" / etc.
    fa = re.search(
        r'(?:final\s*answer|my\s*answer|answer)\s*[:\-]\s*(yes|no)\b',
        t, re.IGNORECASE,
    )
    if fa:
        return 'Yes' if fa.group(1).lower() == 'yes' else 'No'

    # 4. Phrase-based inference. Order matters: check NEGATIVE phrases
    # first because they often contain 'sarcastic' as a substring.
    low = t.lower()
    neg_phrases = [
        'is not being sarcastic', 'not being sarcastic',
        'is not sarcastic', 'not sarcastic',
        'no sarcasm', 'no clear sarcasm', 'is not humorous',
        'not humorous', 'no humor', 'is sincere', 'sincere statement',
        'literal statement', 'taken literally', 'no indication of sarcasm',
    ]
    pos_phrases = [
        'is being sarcastic', 'being sarcastic', 'is sarcastic',
        'sarcastic', 'sarcasm is present', 'indicates sarcasm',
        'suggests sarcasm', 'indicator of sarcasm',
        'is humorous', 'is being humorous', 'humor is present',
    ]
    # Find earliest occurrence of any negative or positive phrase.
    first_neg = min((low.find(p) for p in neg_phrases if low.find(p) != -1), default=-1)
    first_pos = min((low.find(p) for p in pos_phrases if low.find(p) != -1), default=-1)
    if first_neg != -1 and (first_pos == -1 or first_neg <= first_pos):
        return 'No'
    if first_pos != -1:
        return 'Yes'

    return ''


# =============================================================================
# Instance runner
# =============================================================================

def run_instance(test_file, dataset, dataset_name, output_file,
                 system_prompt, model):
    try:
        config = get_dataset_config(dataset_name)
        sample = dataset[test_file]
        target_text = config.get_text_field(sample)
        label = config.get_label_field(sample)
        query = config.get_task_query()

        audio_path = get_audio_path(test_file, dataset_name)
        video_path = get_muted_video_path(test_file, dataset_name)
        if not os.path.exists(audio_path):
            audio_path = None
        if not os.path.exists(video_path):
            video_path = None

        messages = _build_messages(
            system_prompt, query, target_text, audio_path, video_path,
        )

        start_time = time.time()
        final_answer, usage = _call_llm(messages, model)
        elapsed = time.time() - start_time

        verdict = _extract_verdict(final_answer or '')
        print(
            f'[{test_file}] {elapsed:.1f}s | '
            f'tok={usage.get("total_tokens", 0)} | '
            f'verdict={verdict or "?"} | '
            f'{(final_answer or "")[:60]}'
        )

        result = {
            test_file: {
                'answer': [final_answer or ''],
                'parsed_answer': [verdict],
                'label': label,
                'method': 'pure_gemini_single_call',
                'latency': elapsed,
                'api_calls': usage.get('api_calls', 0),
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'had_audio': audio_path is not None,
                'had_video': video_path is not None,
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
# Parallel runner
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


def run_parallel(dataset, dataset_name, max_workers, output_file, resume,
                 system_prompt, model):
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
    print(f'Pure Gemini single-call | Model: {model}')
    print(f'Samples: {len(test_list)} | Workers: {max_workers}')
    print(f'Output: {output_file}')
    print('=' * 60)

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_instance, t, dataset, dataset_name, output_file,
                system_prompt, model,
            ): t for t in test_list
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
    n = 0
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

    if n == 0:
        return

    cost = total_pt / 1e6 * 0.075 + total_ct / 1e6 * 0.30
    print(f'\n{"=" * 60}')
    print(f'PURE GEMINI PERFORMANCE & COST SUMMARY')
    print(f'{"=" * 60}')
    print(f'Samples:                  {n}')
    print(f'Wall-clock:               {wall_time:.1f}s')
    print('-' * 40)
    print(f'Total API Calls:          {total_api}')
    print(f'Avg API Calls/Sample:     {total_api / n:.1f}')
    print(f'Avg Latency/Sample:       {sum(latencies) / n:.2f}s')
    print('-' * 40)
    print(f'Total Prompt Tokens:      {total_pt:,}')
    print(f'Total Completion Tokens:  {total_ct:,}')
    print(f'Total Tokens:             {total_tok:,}')
    print(f'Avg Tokens/Sample:        {total_tok / n:,.0f}')
    print('-' * 40)
    print(f'Estimated Cost:           ${cost:.4f}')
    print(f'Avg Cost/Sample:          ${cost / n:.6f}')
    print(f'{"=" * 60}\n')


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pure Gemini single-call baseline — one model call per '
                    'sample with all three modalities at once, prompts '
                    'built entirely from the CTM config.',
    )
    parser.add_argument('--dataset_name', type=str, default='mustard',
                        choices=['urfunny', 'mustard'])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    system_prompt = _build_system_prompt(args.dataset_name)
    output_file = (
        args.output or f'pure_gemini_{args.dataset_name}.jsonl'
    )

    dataset = load_data(args.dataset)
    print(f'Dataset: {args.dataset_name} | Model: {args.model} | '
          f'Samples: {len(dataset)}')
    print(f'System prompt length: {len(system_prompt)} chars')

    run_parallel(
        dataset, args.dataset_name,
        max_workers=args.max_workers, output_file=output_file,
        resume=args.resume, system_prompt=system_prompt, model=args.model,
    )
