"""
Baseline: Qwen3-VL-thinking with video + text input.

Falls back to frames if video API returns too-long/too-short error.

Examples:
python run_qwen3vl_thinking_baseline.py --dataset_name urfunny --output baseline_urfunny_qwen3vl_thinking.jsonl
python run_qwen3vl_thinking_baseline.py --dataset_name mustard --output baseline_mustard_qwen3vl_thinking.jsonl
"""

import argparse
import base64
import json
import os
import sys
import time

import litellm
from dataset_configs import get_dataset_config
from llm_utils import load_data, load_processed_keys, normalize_label, save_result_to_jsonl

sys.path.append('..')

MODEL = 'qwen/qwen3-vl-8b-thinking'
OPENAI_MODEL = 'openai/qwen3-vl-8b-thinking'
API_BASE = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
THINKING_BUDGET = 4096
MAX_FRAMES = 8


def get_video_path(test_file, dataset_name):
    config = get_dataset_config(dataset_name)
    data_paths = config.get_data_paths()
    base_dir = data_paths['video_only']
    filename = config.get_video_filename(test_file, 'muted')
    return os.path.join(base_dir, filename)


def build_video_message(query, video_path):
    """Build message with video_url for Qwen VL."""
    with open(video_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': query},
            {'type': 'video_url', 'video_url': {'url': f'data:video/mp4;base64,{encoded}'}},
        ]}
    ]


def build_frames_message(query, video_path, max_frames=MAX_FRAMES):
    """Extract frames from video and build message with images."""
    import cv2
    import tempfile

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = [int(total * i / max_frames) for i in range(max_frames)]
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            _, buf = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buf).decode('utf-8'))
        idx += 1
    cap.release()

    if not frames:
        return None

    content = [{'type': 'text', 'text': f'{query}\n\nBelow are {len(frames)} frames from the video.'}]
    for f in frames:
        content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{f}'}})

    return [{'role': 'user', 'content': content}]


def call_model(messages, max_retries=3):
    """Call Qwen3-VL-thinking with retry and frames fallback."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = litellm.completion(
                model=OPENAI_MODEL,
                api_base=API_BASE,
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                extra_body={'thinking_budget': THINKING_BUDGET},
            )
            text = resp.choices[0].message.content
            usage = {
                'prompt_tokens': resp.usage.prompt_tokens,
                'completion_tokens': resp.usage.completion_tokens,
            }
            return text, usage
        except Exception as e:
            err = str(e)
            if 'too long' in err or 'too short' in err:
                return None, None  # Signal to use frames
            print(f'  Attempt {attempt}/{max_retries} failed: {e}')
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None, {'prompt_tokens': 0, 'completion_tokens': 0}


def run_instance(test_file, dataset, dataset_name, output_file):
    config = get_dataset_config(dataset_name)
    sample = dataset[test_file]

    target = config.get_text_field(sample)
    system_prompt = config.get_system_prompt()
    label = config.get_label_field(sample)
    video_path = get_video_path(test_file, dataset_name)

    query = f"{system_prompt}\n\ntarget text: '{target}'"
    print(f'[{test_file}] target: {target[:80]}...')

    start = time.time()

    # Try video first
    use_frames = False
    if os.path.exists(video_path):
        messages = build_video_message(query, video_path)
        answer, usage = call_model(messages)
        if answer is None and usage is None:
            # Video too long/short, fallback to frames
            print(f'[{test_file}] Video error, falling back to frames')
            use_frames = True

    if use_frames or not os.path.exists(video_path):
        if os.path.exists(video_path):
            messages = build_frames_message(query, video_path)
        else:
            messages = [{'role': 'user', 'content': query}]
        if messages:
            answer, usage = call_model(messages)
        else:
            answer, usage = None, {'prompt_tokens': 0, 'completion_tokens': 0}

    duration = time.time() - start
    print(f'[{test_file}] answer: {(answer or "None")[:80]}... ({duration:.1f}s)')

    result = {
        test_file: {
            'answer': [answer],
            'label': label,
            'label_normalized': normalize_label(label),
            'method': 'qwen3vl_thinking_baseline',
            'used_frames': use_frames,
            'usage': usage or {},
            'latency': duration,
        }
    }
    save_result_to_jsonl(result, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='urfunny', choices=['urfunny', 'mustard'])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = args.output or f'baseline_{args.dataset_name}_qwen3vl_thinking.jsonl'

    litellm.set_verbose = False
    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    processed = load_processed_keys(output_file)
    if processed:
        print(f'Resuming: {len(processed)} done, {len(test_list) - len(processed)} remaining')

    print(f'Dataset: {args.dataset_name} | Model: {MODEL} | Thinking: {THINKING_BUDGET}')

    for test_file in test_list:
        if test_file in processed:
            continue
        try:
            run_instance(test_file, dataset, args.dataset_name, output_file)
        except Exception as e:
            print(f'[ERROR] {test_file}: {e}')
        time.sleep(1)
