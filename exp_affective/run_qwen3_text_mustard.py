"""
Run Qwen3-8B (text-only, reasoning mode) on MUStARD sarcasm detection baseline.
Uses the same prompt as CTM's language_processor.

Usage:
    python run_qwen3_text_mustard.py --output results_qwen3_text_mustard.jsonl
    python run_qwen3_text_mustard.py --thinking-budget 0 --output results_qwen3_nothink.jsonl
"""

import argparse
import json
import os
import time

from openai import OpenAI

from dataset_configs import get_dataset_config
from llm_utils import load_data, load_processed_keys

QWEN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Same as processor_language.py default
SYSTEM_PROMPT = "You are an expert in language understanding. Your task is to analyze the provided text and answer questions about it."


def call_model(client, model, query, text, thinking_budget=4096, max_retries=3):
    """Call Qwen3 with text-only, matching CTM language_processor format.
    Uses streaming when thinking is enabled (DashScope requires stream=True for enable_thinking).
    """
    user_message = f"{query}\n The relevant text of the query is: {text}\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            if thinking_budget > 0:
                # Streaming required for thinking mode
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    extra_body={"enable_thinking": True, "thinking_budget": thinking_budget},
                )
                answer = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        answer += delta.content
                # Token usage not available in streaming; estimate from lengths
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
            else:
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
            print(f"  Error attempt {attempt}/{max_retries}: {e}")

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

        print(f"\n[{i+1}/{len(test_list)}] ID={test_file} label={label}")
        print(f"  Text: {text[:100]}...")

        start = time.time()
        answer, usage = call_model(
            client, args.model, task_query, text,
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
    parser = argparse.ArgumentParser(description="Qwen3 text-only sarcasm detection on MUStARD (with reasoning)")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default="results_qwen3_text_mustard.jsonl")
    parser.add_argument("--model", type=str, default="qwen3-8b", help="Model name on DashScope")
    parser.add_argument("--thinking-budget", type=int, default=4096, help="Thinking token budget (0 to disable)")
    args = parser.parse_args()
    run(args)
