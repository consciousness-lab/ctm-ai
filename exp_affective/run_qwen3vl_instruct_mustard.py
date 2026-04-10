"""
Run Qwen3-VL-8B-Instruct on MUStARD sarcasm detection - text + video baseline.
No thinking budget (instruct model).

Usage:
    python run_qwen3vl_instruct_mustard.py --output results_qwen3vl_instruct_mustard.jsonl
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
    query_text = f"{query} Answer with Yes or No first, then explain briefly.\n The relevant text of the query is: {text}\n"
    user_content = [
        {"type": "text", "text": query_text},
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

    video_failed = False
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            answer = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            if answer:
                return answer, usage
            print(f"  Warning: empty response, attempt {attempt}/{max_retries}")
        except Exception as e:
            err_str = str(e)
            print(f"  Error attempt {attempt}/{max_retries}: {e}")
            if any(kw in err_str for kw in ['too long', 'too short', 'InvalidParameter', 'DataInspectionFailed']):
                video_failed = True
                break

    # Fallback 1: try frames if video caused errors
    if video_path and video_failed:
        print("  Fallback: retrying with extracted frames...")
        try:
            frames = extract_frames_base64(video_path)
            if frames:
                frames_content = [{"type": "text", "text": query_text}]
                for fb64 in frames:
                    frames_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
                    })
                frames_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": frames_content},
                ]
                for attempt in range(1, max_retries + 1):
                    try:
                        response = client.chat.completions.create(model=model, messages=frames_messages)
                        answer = response.choices[0].message.content
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        }
                        if answer:
                            return answer, usage
                        print(f"  Warning: empty frames response, attempt {attempt}/{max_retries}")
                    except Exception as e:
                        print(f"  Frames error attempt {attempt}/{max_retries}: {e}")
        except Exception as e:
            print(f"  Frame extraction failed: {e}")

    # Fallback 2: retry with text-only
    if video_path:
        print("  Fallback: retrying with text only...")
        text_only_content = [
            {"type": "text", "text": query_text},
        ]
        fallback_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text_only_content},
        ]
        try:
            response = client.chat.completions.create(model=model, messages=fallback_messages)
            answer = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            if answer:
                return answer, usage
        except Exception as e:
            print(f"  Text-only fallback also failed: {e}")

    return None, {"prompt_tokens": 0, "completion_tokens": 0}


def parse_prediction(answer):
    if not answer:
        return None
    lower = answer.strip().lower()
    if lower.startswith("yes"):
        return 1
    elif lower.startswith("no"):
        return 0
    if "not sarcastic" in lower or "not being sarcastic" in lower or "no sarcasm" in lower or "is not sarcastic" in lower:
        return 0
    if "is sarcastic" in lower or "is being sarcastic" in lower or "being sarcastic" in lower or "sarcasm" in lower:
        return 1
    return None


def run(args):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=QWEN_API_BASE,
    )

    config = get_dataset_config("mustard")
    dataset_path = args.dataset or config.get_default_dataset_path()
    dataset = load_data(dataset_path)
    test_list = list(dataset.keys())
    task_query = config.get_task_query()
    data_paths = config.get_data_paths()

    done_ids = load_processed_keys(args.output)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done, skipping.")

    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_path} ({len(test_list)} samples)")
    print(f"Task query: {task_query}")
    print(f"Output: {args.output}")
    print("=" * 60)

    correct = 0
    total = 0
    total_time = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, test_file in enumerate(test_list):
        if test_file in done_ids:
            continue

        sample = dataset[test_file]
        label = config.get_label_field(sample)
        text = config.get_context_field(sample)

        video_filename = config.get_video_filename(test_file, "full")
        video_path = os.path.join(data_paths["full_video"], video_filename)
        if not os.path.exists(video_path):
            video_path = None

        print(f"\n[{i+1}/{len(test_list)}] ID={test_file} label={label}")
        print(f"  Text: {text[:100]}...")
        print(f"  Video: {'yes' if video_path else 'missing'}")

        start = time.time()
        answer, usage = call_model(
            client, args.model, task_query, text,
            video_path=video_path,
        )
        elapsed = time.time() - start
        total_time += elapsed
        total_input_tokens += usage["prompt_tokens"]
        total_output_tokens += usage["completion_tokens"]

        pred = parse_prediction(answer)
        is_correct = (pred == label) if pred is not None else False
        if is_correct:
            correct += 1
        total += 1

        print(f"  Answer: {answer[:120] if answer else 'None'}...")
        print(f"  Pred={pred} Label={label} Correct={is_correct} Time={elapsed:.1f}s")

        result = {
            test_file: {
                "answer": answer[:500] if answer else None,
                "label": label,
                "pred": pred,
                "correct": is_correct,
                "time": round(elapsed, 1),
                "tokens": usage,
                "has_video": video_path is not None,
            }
        }

        with open(args.output, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        acc = correct / total * 100
        print(f"  Running accuracy: {correct}/{total} = {acc:.1f}%")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"Total time: {total_time:.1f}s  Avg: {total_time/total:.1f}s/sample")
        print(f"Total tokens: {total_input_tokens} in / {total_output_tokens} out")
    else:
        print("No new samples processed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL-Instruct sarcasm detection on MUStARD")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSON (default: mustard test)")
    parser.add_argument("--output", type=str, default="results_qwen3vl_instruct_mustard.jsonl")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-instruct", help="Model name on DashScope")
    args = parser.parse_args()
    run(args)
