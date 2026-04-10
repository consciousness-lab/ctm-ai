"""
Run Qwen3-VL-8B-Instruct on URFunny humor detection - text + video baseline.
Falls back to frames if video API errors.

Usage:
    python run_qwen3vl_instruct_urfunny.py --output results_qwen3vl_instruct_urfunny.jsonl
"""

import argparse
import base64
import json
import os
import time

from openai import OpenAI

from dataset_configs import get_dataset_config
from llm_utils import load_data, load_processed_keys

QWEN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = (
    "You are an expert in language and video understanding. "
    "Your task is to analyze the provided text and video, and answer questions about them."
)


def encode_video_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_frames_base64(video_path, max_frames=8):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(total * i / max_frames) for i in range(max_frames)]
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            _, buf = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buf).decode())
        idx += 1
    cap.release()
    return frames


def call_model(client, model, query, text, video_path=None, max_retries=3):
    user_content = [
        {"type": "text", "text": f"{query}\nAnswer with Yes or No first, then explain briefly.\n The relevant text of the query is: {text}\n"},
    ]

    if video_path and os.path.exists(video_path):
        encoded = encode_video_base64(video_path)
        user_content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
        })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            answer = response.choices[0].message.content
            usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
            if answer:
                return answer, usage
        except Exception as e:
            err = str(e)
            if 'too long' in err or 'too short' in err or 'InvalidParameter' in err or 'DataInspectionFailed' in err:
                break
            print(f"  Error attempt {attempt}/{max_retries}: {e}")

    # Fallback: frames
    if video_path and os.path.exists(video_path):
        print("  Fallback: trying frames...")
        frames = extract_frames_base64(video_path)
        if frames:
            frame_content = [{"type": "text", "text": f"{query}\nAnswer with Yes or No first, then explain briefly.\n The relevant text of the query is: {text}\n"}]
            for f in frames:
                frame_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
            try:
                response = client.chat.completions.create(model=model, messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": frame_content},
                ])
                answer = response.choices[0].message.content
                if answer:
                    return answer, {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
            except Exception as e:
                print(f"  Frames failed: {e}")

    # Last fallback: text only
    print("  Fallback: text only...")
    try:
        response = client.chat.completions.create(model=model, messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{query}\nAnswer with Yes or No first, then explain briefly.\n The relevant text of the query is: {text}\n"},
        ])
        answer = response.choices[0].message.content
        if answer:
            return answer, {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
    except Exception as e:
        print(f"  Text-only failed: {e}")

    return None, {"prompt_tokens": 0, "completion_tokens": 0}


def run(args):
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=QWEN_API_BASE)

    config = get_dataset_config("urfunny")
    dataset_path = args.dataset or config.get_default_dataset_path()
    dataset = load_data(dataset_path)
    test_list = list(dataset.keys())
    task_query = config.get_task_query()
    data_paths = config.get_data_paths()

    done_ids = load_processed_keys(args.output)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done.")

    print(f"Model: {args.model} | Dataset: urfunny ({len(test_list)} samples) | Output: {args.output}")

    for i, test_file in enumerate(test_list):
        if test_file in done_ids:
            continue

        sample = dataset[test_file]
        label = config.get_label_field(sample)
        text = config.get_text_field(sample)

        video_filename = config.get_video_filename(test_file, "muted")
        video_path = os.path.join(data_paths["video_only"], video_filename)
        if not os.path.exists(video_path):
            video_path = None

        start = time.time()
        answer, usage = call_model(client, args.model, task_query, text, video_path=video_path)
        elapsed = time.time() - start

        print(f"[{i+1}/{len(test_list)}] {test_file}: {(answer or 'None')[:80]}... ({elapsed:.1f}s)")

        result = {test_file: {"answer": answer, "label": label, "time": round(elapsed, 1), "tokens": usage}}
        with open(args.output, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default="results_qwen3vl_instruct_urfunny.jsonl")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-instruct")
    args = parser.parse_args()
    run(args)
