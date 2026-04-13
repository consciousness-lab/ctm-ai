"""
MoA (Mixture of Agents) baseline for affective detection (MUStARD / URFunny).

This baseline faithfully reproduces the Mixture-of-Agents algorithm from the
original paper (`MoA/advanced-moa.py`, `MoA/utils.py`) with a multimodal
adaptation. It imports `inject_references_to_messages` directly from the
original MoA source tree — we do NOT reimplement the aggregation logic.

Algorithm (multi-layer, matches the "Advanced MoA" pattern):
    Layer 1 (proposers):
        3 modality proposers run in parallel on their own modality input.
        text proposer   → reads dialogue text
        audio proposer  → reads audio file
        video proposer  → reads muted video
        No references are injected at this layer.

    Layer 2..L (recursive refinement):
        Each proposer runs again on its own modality input, but with the
        previous layer's outputs injected as "references" into the system
        prompt via `inject_references_to_messages`. This lets each modality
        expert see what the other experts concluded and refine its own
        analysis (the core MoA contribution).

    Final aggregator:
        A final model takes the last layer's proposer outputs as references
        and produces the Yes/No verdict. `inject_references_to_messages` is
        used again to build the aggregator's system prompt, exactly as in
        `MoA/advanced-moa.py`.

The original MoA paper uses 3 layers. We default to 2 (one initial layer +
one refinement) to keep API cost reasonable for multimodal inputs, but
`--layers 3` matches the paper exactly.

Because the original MoA proposers are all text-only, we use a single
backbone (gemini-2.5-flash-lite) for all modality proposers by default.
Model diversity can be restored via `--proposer_models` (comma-separated
list), matching the MoA paper's "diverse proposers" setup.

Examples:
    python run_moa.py --dataset_name mustard --layers 2 --max_workers 8
    python run_moa.py --dataset_name urfunny --layers 3 --max_workers 4
    python run_moa.py --dataset_name mustard --resume
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- Import original MoA source ---------------------------------------------
# MoA is not a pip package; we import its utils.py directly from the repo
# sibling checkout so that the core aggregation primitive
# `inject_references_to_messages` is *literally* the one from the paper.
_MOA_REPO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'MoA',
))
if _MOA_REPO not in sys.path:
    sys.path.insert(0, _MOA_REPO)
from utils import inject_references_to_messages  # noqa: E402  (MoA original source)

# --- Project imports --------------------------------------------------------
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
DEFAULT_LAYERS = 2  # layer 1 = initial; layer 2 = refinement; then aggregator
MODALITIES = ('text', 'audio', 'video')


# =============================================================================
# Proposer system prompts — loaded from CTM config so we share them with the
# other baselines (MetaGPT, AutoGen). This keeps the architectural comparison
# fair: only the *orchestration* differs between baselines.
# =============================================================================

def _load_ctm_prompts(dataset_name='mustard'):
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
        cfg = json.load(f)
    return cfg['processors_config']


def _get_aggregator_prompt(dataset_name='mustard'):
    """Final aggregator system-prompt prefix.

    `inject_references_to_messages` from MoA/utils.py will append the text
    'Responses from models:' followed by a numbered list of reference
    responses to whatever system prompt we pass in. So this prompt is the
    task-specific prefix that precedes the injected references.
    """
    if dataset_name == 'urfunny':
        yes_label, no_label = 'humorous', 'not humorous'
    else:
        yes_label, no_label = 'sarcastic', 'not sarcastic'
    return (
        f'You are the final aggregator in a Mixture-of-Agents pipeline for '
        f'detecting whether a person is {yes_label}. You will receive '
        f'analyses from modality-specific experts (text, audio, video). '
        f'Critically synthesize them and produce a single definitive answer.\n\n'
        f'Guidelines:\n'
        f'- Weigh evidence across modalities; text usually gives the strongest '
        f'semantic signal, audio gives tonal confirmation, video gives '
        f'supplementary cues.\n'
        f'- If experts disagree, decide which has the strongest evidence.\n'
        f'- Do not just follow the majority — weigh reasoning quality.\n\n'
        f'Your answer MUST start with exactly "Yes" ({yes_label}) or "No" '
        f'({no_label}), followed by a brief justification citing the most '
        f'decisive evidence.'
    )


# Module-level defaults (overridden from main() once --dataset_name is known)
_CTM_PROMPTS = _load_ctm_prompts()
TEXT_PROPOSER_SYSTEM = _CTM_PROMPTS['language_processor']['system_prompt']
AUDIO_PROPOSER_SYSTEM = _CTM_PROMPTS['audio_processor']['system_prompt']
VIDEO_PROPOSER_SYSTEM = _CTM_PROMPTS['video_processor']['system_prompt']
AGGREGATOR_SYSTEM = _get_aggregator_prompt()


# =============================================================================
# LLM call (shared by proposers and aggregator)
# =============================================================================

def _call_llm(messages, model, max_retries=3):
    """Single litellm.completion call with retry + hard per-request timeout
    so the whole run doesn't wedge when a single Gemini multimodal request
    stalls (intermittent failure mode observed on this dataset). Returns
    (text, usage)."""
    start = time.time()
    api_calls = 0
    per_request_timeout = 90
    for attempt in range(1, max_retries + 1):
        try:
            api_calls += 1
            resp = litellm.completion(
                model=model, messages=messages, temperature=0.0,
                timeout=per_request_timeout,
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
# Proposer message builders (per modality)
# =============================================================================

def _build_text_messages(query, target_sentence, references=None):
    """Build text proposer messages. References (from previous layer) are
    injected via the MoA primitive when provided."""
    messages = [
        {'role': 'system', 'content': TEXT_PROPOSER_SYSTEM},
        {'role': 'user', 'content': (
            f'{query}\n\nThe relevant text of the query is: {target_sentence}'
        )},
    ]
    if references:
        messages = inject_references_to_messages(messages, references)
    return messages


def _build_audio_messages(query, audio_path, references=None):
    """Build audio proposer messages; returns None if audio is unavailable."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    ext = audio_path.split('.')[-1].lower()
    mime_map = {'mp3': 'audio/mp3', 'wav': 'audio/wav',
                'mp4': 'audio/mp4', 'aac': 'audio/aac'}
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
    if references:
        messages = inject_references_to_messages(messages, references)
    return messages


def _build_video_messages(query, video_path, references=None):
    """Build video proposer messages; returns None if video is unavailable."""
    if not video_path or not os.path.exists(video_path):
        return None
    with open(video_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    messages = [
        {'role': 'system', 'content': VIDEO_PROPOSER_SYSTEM},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': f'{query}\n\nAnalyze the muted video.'},
            {'type': 'image_url', 'image_url': {'url': f'data:video/mp4;base64,{encoded}'}},
        ]},
    ]
    if references:
        messages = inject_references_to_messages(messages, references)
    return messages


def _run_modality(modality, query, target_sentence, audio_path, video_path,
                  references, model):
    """Dispatch one modality proposer call. Returns (text, usage)."""
    if modality == 'text':
        msgs = _build_text_messages(query, target_sentence, references)
    elif modality == 'audio':
        msgs = _build_audio_messages(query, audio_path, references)
    elif modality == 'video':
        msgs = _build_video_messages(query, video_path, references)
    else:
        raise ValueError(f'Unknown modality: {modality}')

    if msgs is None:
        empty = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
                 'api_calls': 0, 'latency': 0.0}
        return f'[{modality} unavailable]', empty

    return _call_llm(msgs, model=model)


# =============================================================================
# Multi-layer MoA pipeline
# =============================================================================

def _run_proposer_layer(query, target_sentence, audio_path, video_path,
                        references, proposer_models, layer_idx):
    """Run all 3 modality proposers in parallel for one layer. Returns
    ({modality -> text}, usages_list)."""
    results = {}
    usages = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _run_modality, m, query, target_sentence,
                audio_path, video_path, references, proposer_models[m],
            ): m for m in MODALITIES
        }
        for fut in as_completed(futures):
            m = futures[fut]
            text, usage = fut.result()
            results[m] = text or f'[layer{layer_idx} {m} failed]'
            usages.append((f'layer{layer_idx}_{m}', usage))
    return results, usages


def run_moa_pipeline(query, target_sentence, audio_path, video_path,
                     layers, proposer_models, aggregator_model):
    """Execute L-layer MoA for one sample.

    proposer_models: dict {'text': model, 'audio': model, 'video': model}
                     (allows mixing models per modality — matches MoA's
                     diverse-model design when multiple models are supplied).
    """
    all_usages = []
    layer_results = [None] * layers  # list of {modality -> text}

    # ---- Layer 1: initial proposers (no references) ----
    layer_results[0], usages = _run_proposer_layer(
        query, target_sentence, audio_path, video_path,
        references=None, proposer_models=proposer_models, layer_idx=1,
    )
    all_usages.extend(usages)

    # ---- Layers 2..L: refinement with references from previous layer ----
    for layer_idx in range(2, layers + 1):
        prev = layer_results[layer_idx - 2]
        # References passed to inject_references_to_messages: tag each with
        # the modality so the LLM can tell them apart in the numbered list.
        refs = [f'[{m.capitalize()} expert] {prev[m]}' for m in MODALITIES]
        layer_results[layer_idx - 1], usages = _run_proposer_layer(
            query, target_sentence, audio_path, video_path,
            references=refs, proposer_models=proposer_models, layer_idx=layer_idx,
        )
        all_usages.extend(usages)

    # ---- Final aggregator: synthesize last layer's outputs ----
    final_refs = [
        f'[{m.capitalize()} expert] {layer_results[-1][m]}'
        for m in MODALITIES
    ]
    agg_messages = [
        {'role': 'system', 'content': AGGREGATOR_SYSTEM},
        {'role': 'user', 'content': query},
    ]
    agg_messages = inject_references_to_messages(agg_messages, final_refs)
    final_answer, agg_usage = _call_llm(agg_messages, model=aggregator_model)
    all_usages.append(('aggregator', agg_usage))

    # ---- Aggregate usage stats ----
    usage_stats = {
        'total_api_calls': sum(u.get('api_calls', 0) for _, u in all_usages),
        'total_prompt_tokens': sum(u.get('prompt_tokens', 0) for _, u in all_usages),
        'total_completion_tokens': sum(u.get('completion_tokens', 0) for _, u in all_usages),
        'total_tokens': sum(u.get('total_tokens', 0) for _, u in all_usages),
        'stage_latencies': {name: u.get('latency', 0.0) for name, u in all_usages},
        'layers': layers,
    }

    stage_outputs = {
        f'layer{layer_idx+1}_{m}': layer_results[layer_idx][m]
        for layer_idx in range(layers)
        for m in MODALITIES
    }

    return final_answer, stage_outputs, usage_stats


# =============================================================================
# Instance runner
# =============================================================================

def run_instance(test_file, dataset, dataset_name, output_file, layers,
                 proposer_models, aggregator_model):
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
            layers=layers, proposer_models=proposer_models,
            aggregator_model=aggregator_model,
        )
        elapsed = time.time() - start_time

        parsed = final_answer.strip() if final_answer else ''
        print(
            f'[{test_file}] {elapsed:.1f}s | L={layers} | '
            f'api={usage_stats["total_api_calls"]} '
            f'tok={usage_stats["total_tokens"]} | {parsed[:60]}'
        )

        result = {
            test_file: {
                'answer': [final_answer or ''],
                'parsed_answer': [parsed],
                'label': label,
                'method': f'moa_{layers}layer',
                'latency': elapsed,
                'api_calls': usage_stats['total_api_calls'],
                'prompt_tokens': usage_stats['total_prompt_tokens'],
                'completion_tokens': usage_stats['total_completion_tokens'],
                'total_tokens': usage_stats['total_tokens'],
                'stage_latencies': usage_stats['stage_latencies'],
                'stage_outputs': stage_outputs,
                'layers': layers,
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
                 sleep_between, layers, proposer_models, aggregator_model):
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
    print(f'MoA {layers}-layer | Proposers: {proposer_models} | '
          f'Aggregator: {aggregator_model}')
    print(f'Samples: {len(test_list)} | Workers: {max_workers}')
    print(f'Output: {output_file}')
    print('=' * 60)

    start_time = time.time()
    completed = 0

    if max_workers == 1 and sleep_between > 0:
        for t in test_list:
            run_instance(
                t, dataset, dataset_name, output_file,
                layers, proposer_models, aggregator_model,
            )
            completed += 1
            print(f'Progress: {completed}/{len(test_list)}')
            if completed < len(test_list):
                time.sleep(sleep_between)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_instance, t, dataset, dataset_name, output_file,
                    layers, proposer_models, aggregator_model,
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
    _print_summary(output_file, wall_time, layers)


def _print_summary(output_file, wall_time, layers):
    total_api = total_pt = total_ct = total_tok = 0
    latencies = []
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
    except Exception as e:
        print(f'Warning: {e}')
        return

    if n == 0:
        return

    cost = total_pt / 1e6 * 0.075 + total_ct / 1e6 * 0.30
    print(f'\n{"=" * 60}')
    print(f'MOA {layers}-LAYER PERFORMANCE & COST SUMMARY')
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

def _parse_proposer_models(spec, fallback):
    """Parse --proposer_models.

    Accepted forms:
        <single model>                 → used for all 3 modalities
        <text>,<audio>,<video>         → positional
        text=<m>,audio=<m>,video=<m>   → explicit keys (any subset)
    """
    if not spec:
        return {m: fallback for m in MODALITIES}
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    if len(parts) == 1:
        return {m: parts[0] for m in MODALITIES}
    models = {m: fallback for m in MODALITIES}
    if '=' in parts[0]:
        for part in parts:
            k, v = part.split('=', 1)
            if k not in MODALITIES:
                raise ValueError(f'Unknown modality: {k}')
            models[k] = v.strip()
    else:
        if len(parts) != 3:
            raise ValueError(
                'Positional --proposer_models needs exactly 3 comma-separated '
                'models (text,audio,video)'
            )
        for m, v in zip(MODALITIES, parts):
            models[m] = v
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-layer MoA baseline for affective detection. '
                    'Uses inject_references_to_messages from the original MoA '
                    'source tree (see MoA/utils.py).',
    )
    parser.add_argument('--dataset_name', type=str, default='mustard',
                        choices=['urfunny', 'mustard'])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sleep', type=float, default=0)
    parser.add_argument('--layers', type=int, default=DEFAULT_LAYERS,
                        help='Number of MoA proposer layers (>=1). '
                             'Paper default is 3.')
    parser.add_argument('--proposer_models', type=str, default=None,
                        help='Comma-separated proposer models. Either one '
                             'model for all modalities, or exactly three in '
                             'text,audio,video order, or explicit key=model '
                             'pairs (e.g. text=gemini/gemini-2.5-pro,'
                             'audio=gemini/gemini-2.5-flash-lite). '
                             f'Default: {DEFAULT_MODEL} for all.')
    parser.add_argument('--aggregator_model', type=str, default=DEFAULT_MODEL,
                        help='Final aggregator model (MoA paper uses a more '
                             'capable aggregator than proposers).')
    args = parser.parse_args()

    if args.layers < 1:
        parser.error('--layers must be >= 1')

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()

    # Reload prompts for the target dataset
    _this = sys.modules[__name__]
    _prompts = _load_ctm_prompts(args.dataset_name)
    _this.TEXT_PROPOSER_SYSTEM = _prompts['language_processor']['system_prompt']
    _this.AUDIO_PROPOSER_SYSTEM = _prompts['audio_processor']['system_prompt']
    _this.VIDEO_PROPOSER_SYSTEM = _prompts['video_processor']['system_prompt']
    _this.AGGREGATOR_SYSTEM = _get_aggregator_prompt(args.dataset_name)

    proposer_models = _parse_proposer_models(args.proposer_models, DEFAULT_MODEL)

    output_file = (
        args.output or f'moa_{args.dataset_name}_{args.layers}layer.jsonl'
    )

    dataset = load_data(args.dataset)
    print(f'Dataset: {args.dataset_name} | Samples: {len(dataset)}')
    print(f'Layers:  {args.layers}')
    print(f'Proposers: {proposer_models}')
    print(f'Aggregator: {args.aggregator_model}')

    run_parallel(
        dataset, args.dataset_name,
        max_workers=args.max_workers, output_file=output_file,
        resume=args.resume, sleep_between=args.sleep,
        layers=args.layers, proposer_models=proposer_models,
        aggregator_model=args.aggregator_model,
    )
