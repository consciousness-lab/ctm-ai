"""
Run Qwen3-VL (reasoning mode) on MUStARD sarcasm detection - text + video baseline.
Combines language_processor + video_processor prompts from CTM.
Uses openai SDK directly (litellm doesn't forward thinking_budget correctly).

Usage:
    python run_qwen3vl_mustard.py --output results_qwen3vl_mustard.jsonl
    python run_qwen3vl_mustard.py --thinking-budget 0 --output results_qwen3vl_nothink.jsonl
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

# Combined from processor_language.py + processor_video.py defaults
SYSTEM_PROMPT = (
    "You are an expert in language and video understanding. "
    "Your task is to analyze the provided text and video, and answer questions about them. "
    "Be concise. Answer with Yes or No first, then give a brief explanation in 2-3 sentences."
)

# Few-shot examples to prevent overthinking and calibrate Yes/No threshold.
# Text-only (no video) but demonstrates the expected concise response style.
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Is the person being sarcastic or not?\n"
            " The relevant text of the query is: "
            "Well, we ended up at the diner drinking coffee, and she said the funniest thing.\n"
            "She said she wanted to have a baby with me.\n"
            "That is funny.\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Yes. The response \"That is funny\" is sarcastic because having a baby is a serious "
            "topic, and calling it \"funny\" dismisses it with ironic understatement, implying the "
            "speaker finds the idea absurd rather than genuinely amusing."
        ),
    },
    {
        "role": "user",
        "content": (
            "Is the person being sarcastic or not?\n"
            " The relevant text of the query is: "
            "You know what, I am not even sorry. I am glad we lied because it was a lot of fun.\n"
            "You know what? Me too.\n"
            "Me too. And I would do it again in a heartbeat.\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "No. The speakers are being genuine. They are sincerely expressing that they enjoyed "
            "the experience and would repeat it, with no indication of meaning the opposite."
        ),
    },
    {
        "role": "user",
        "content": (
            "Is the person being sarcastic or not?\n"
            " The relevant text of the query is: "
            "What are you doing?\n"
            "Making chocolate milk. You want some?\n"
            "No thanks, I'm 29.\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Yes. Saying \"I'm 29\" as a reason to decline chocolate milk is sarcastic — it implies "
            "chocolate milk is childish, using dry humor to mock the offer rather than giving a "
            "straightforward refusal."
        ),
    },
    {
        "role": "user",
        "content": (
            "Is the person being sarcastic or not?\n"
            " The relevant text of the query is: "
            "Did you have fun at the bachelor's party last night?\n"
            "Yeah it was great. We had a stripper come and everything.\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "No. The speaker is genuinely describing what happened at the party. The enthusiasm "
            "is straightforward, not ironic — there is no mismatch between the literal words and "
            "the intended meaning."
        ),
    },
]


def encode_video_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(client, model, query, text, video_path=None, thinking_budget=4096, max_retries=3):
    """Call Qwen3-VL with text + video, combining language_processor + video_processor formats."""
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
            print(f"  Warning: empty response, attempt {attempt}/{max_retries}")
        except Exception as e:
            print(f"  Error attempt {attempt}/{max_retries}: {e}")

    # Fallback: retry with text-only if video caused errors
    if video_path:
        print("  Fallback: retrying with text only...")
        text_only_content = [
            {"type": "text", "text": f"{query}\n The relevant text of the query is: {text}\n"},
        ]
        fallback_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": text_only_content},
        ]
        try:
            kwargs = {"model": model, "messages": fallback_messages}
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
            print(f"  Fallback also failed: {e}")

    return None, {"prompt_tokens": 0, "completion_tokens": 0}


def parse_prediction(answer):
    if not answer:
        return None
    lower = answer.strip().lower()
    if lower.startswith("yes"):
        return 1
    elif lower.startswith("no"):
        return 0
    if "is sarcastic" in lower or "is being sarcastic" in lower:
        return 1
    if "not sarcastic" in lower or "not being sarcastic" in lower or "no sarcasm" in lower:
        return 0
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
    print(f"Thinking budget: {args.thinking_budget}")
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
            thinking_budget=args.thinking_budget,
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
    parser = argparse.ArgumentParser(description="Qwen3-VL sarcasm detection on MUStARD")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSON (default: mustard test)")
    parser.add_argument("--output", type=str, default="results_qwen3vl_mustard.jsonl")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-thinking", help="Model name on DashScope")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Thinking token budget (0=use model default, qwen3-vl-8b-thinking has built-in reasoning)")
    args = parser.parse_args()
    run(args)
