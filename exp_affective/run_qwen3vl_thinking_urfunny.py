"""
Run Qwen3-VL-8B-thinking (few-shot) on URFunny humor detection - text + video baseline.
Adapted from run_qwen3vl_mustard.py with humor-specific few-shot examples.
Falls back to frames if video API errors.

Usage:
    python run_qwen3vl_thinking_urfunny.py --output results_qwen3vl_fewshot_urfunny.jsonl
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
    "Your task is to analyze the provided text and video, and answer questions about them. "
    "Be concise. Answer with Yes or No first, then give a brief explanation in 2-3 sentences."
)

# Few-shot examples for humor detection (URFunny = TED talks)
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Is the person being humorous or not?\n"
            " The relevant text of the query is: "
            "for those of you old enough to remember we used to have to go to the store to steal it\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Yes. The speaker uses misdirection humor — the setup implies buying something, "
            "but the punchline reveals 'steal it', subverting expectations for comedic effect."
        ),
    },
    {
        "role": "user",
        "content": (
            "Is the person being humorous or not?\n"
            " The relevant text of the query is: "
            "it has empowered you it has empowered me and it has empowered some other guys as well\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Yes. The speaker uses ironic understatement — after grand statements about empowerment, "
            "'some other guys as well' is a deliberately vague and deflating addition meant to amuse."
        ),
    },
    {
        "role": "user",
        "content": (
            "Is the person being humorous or not?\n"
            " The relevant text of the query is: "
            "the bronze is nice hard durable material that could roll on the ground\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "No. This is a straightforward factual description of a material's properties. "
            "There is no comedic intent, wordplay, or humor technique being used."
        ),
    },
    {
        "role": "user",
        "content": (
            "Is the person being humorous or not?\n"
            " The relevant text of the query is: "
            "today tech companies are the world's largest editors\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "No. This is a serious observation about the role of tech companies in content curation. "
            "While it may be surprising, it is meant as a factual claim, not a joke."
        ),
    },
]


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


def call_model(client, model, query, text, video_path=None, thinking_budget=4096, max_retries=3):
    user_content = [
        {"type": "text", "text": f"{query}\n The relevant text of the query is: {text}\n"},
    ]

    if video_path and os.path.exists(video_path):
        encoded = encode_video_base64(video_path)
        user_content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
        })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"model": model, "messages": messages}
            if thinking_budget > 0:
                kwargs["extra_body"] = {"thinking_budget": thinking_budget}
            response = client.chat.completions.create(**kwargs)
            answer = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            if answer:
                return answer, usage
        except Exception as e:
            err = str(e)
            if 'too long' in err or 'too short' in err or 'InvalidParameter' in err or 'DataInspectionFailed' in err:
                break  # Don't retry, go to frames fallback
            print(f"  Error attempt {attempt}/{max_retries}: {e}")

    # Fallback: try frames
    if video_path and os.path.exists(video_path):
        print("  Fallback: trying frames...")
        frames = extract_frames_base64(video_path)
        if frames:
            frame_content = [{"type": "text", "text": f"{query}\n The relevant text of the query is: {text}\n"}]
            for f in frames:
                frame_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
            frame_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *FEW_SHOT_EXAMPLES,
                {"role": "user", "content": frame_content},
            ]
            try:
                kwargs = {"model": model, "messages": frame_messages}
                if thinking_budget > 0:
                    kwargs["extra_body"] = {"thinking_budget": thinking_budget}
                response = client.chat.completions.create(**kwargs)
                answer = response.choices[0].message.content
                if answer:
                    return answer, {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
            except Exception as e:
                print(f"  Frames also failed: {e}")

    # Last fallback: text only
    print("  Fallback: text only...")
    text_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": [{"type": "text", "text": f"{query}\n The relevant text of the query is: {text}\n"}]},
    ]
    try:
        kwargs = {"model": model, "messages": text_messages}
        if thinking_budget > 0:
            kwargs["extra_body"] = {"thinking_budget": thinking_budget}
        response = client.chat.completions.create(**kwargs)
        answer = response.choices[0].message.content
        if answer:
            return answer, {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
    except Exception as e:
        print(f"  Text-only also failed: {e}")

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
        print(f"Resuming: {len(done_ids)} already done, skipping.")

    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_path} ({len(test_list)} samples)")
    print(f"Output: {args.output}")
    print("=" * 60)

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

        print(f"\n[{i+1}/{len(test_list)}] ID={test_file} label={label}")

        start = time.time()
        answer, usage = call_model(client, args.model, task_query, text, video_path=video_path, thinking_budget=args.thinking_budget)
        elapsed = time.time() - start

        print(f"  Answer: {answer[:100] if answer else 'None'}... ({elapsed:.1f}s)")

        result = {
            test_file: {
                "answer": answer,
                "label": label,
                "time": round(elapsed, 1),
                "tokens": usage,
            }
        }
        with open(args.output, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default="results_qwen3vl_fewshot_urfunny.jsonl")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-thinking")
    parser.add_argument("--thinking-budget", type=int, default=4096)
    args = parser.parse_args()
    run(args)
