"""
Baseline experiment using multimodal input.

Sends full video (with audio) + text (punchline sentence) in a single call per sample.
This is the simplest baseline: no debate, no augmentation, no voting.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import litellm
from llm_utils import (
    add_common_args,
    check_api_key,
    create_agent,
    get_full_video_path,
    load_data,
    load_processed_keys,
    save_result_to_jsonl,
)

sys.path.append("..")

SYS_PROMPT = (
    "Please analyze the inputs provided to determine if the punchline provided is humorous or not."
    "If you think these inputs include exaggerated description or it is expressing humorous meaning, please answer 'Yes'."
    "If you think these inputs are neutral or just common meaning, please answer 'No'. If you are not sure, you should answer 'No'."
    "Your answer should begin with either 'Yes' or 'No', followed by your reasoning."
)

file_lock = Lock()


def run_instance(test_file, dataset, agent, output_file="baseline.jsonl"):
    try:
        print(f"[{test_file}] Starting processing...")

        target_sentence = dataset[test_file]["punchline_sentence"]
        query = "Is the person being humorous or not?"

        # Full video (with audio) from urfunny_videos
        video_path = get_full_video_path(test_file)

        if not os.path.exists(video_path):
            print(f"[{test_file}] Video not exist: {video_path}")
            return f"Video file not found for {test_file}"

        print(f"[{test_file}] Target sentence: {target_sentence[:100]}...")
        print(f"[{test_file}] Video found: {video_path}")

        print(f"[{test_file}] Calling LLM...")
        start_time = time.time()

        # Use MultimodalAgent: full video (with audio) + context (target_sentence)
        answer, usage = agent.call(
            query,
            context=target_sentence,
            video_path=video_path,
        )

        end_time = time.time()

        print(f"[{test_file}] LLM call completed in {end_time - start_time:.2f}s")
        print(f"[{test_file}] Answer: {answer}")

        result = {
            test_file: {
                "answer": [answer],
                "label": dataset[test_file]["label"],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
                "latency": end_time - start_time,
            }
        }

        with file_lock:
            save_result_to_jsonl(result, output_file)

        print(f"[{test_file}] Result saved to {output_file}")
        return f"Successfully processed {test_file}"

    except Exception as e:
        error_msg = f"Error processing {test_file}: {str(e)}"
        print(f"[{test_file}] {error_msg}")
        import traceback

        traceback.print_exc()
        return error_msg


def run_parallel(agent, dataset, output_file, max_workers=4):
    test_list = list(dataset.keys())
    print(f"Total test samples: {len(test_list)}")
    print(f"Using {max_workers} workers")
    print(f"Output file: {output_file}")
    print("=" * 50)

    # Load already processed keys for resume
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f"Resuming: {len(processed_keys)} already processed, skipping...")
        test_list = [t for t in test_list if t not in processed_keys]
        print(f"Remaining: {len(test_list)} samples")

    if not test_list:
        print("All samples already processed!")
        return

    start_time = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(
                run_instance, test_file, dataset, agent, output_file
            ): test_file
            for test_file in test_list
        }

        for future in as_completed(future_to_test):
            test_file = future_to_test[future]
            completed_count += 1

            try:
                result = future.result()
                print(f"Progress: {completed_count}/{len(test_list)} - {result}")
            except Exception as exc:
                print(f"Error processing {test_file}: {exc}")

    end_time = time.time()
    total_time = end_time - start_time
    print("=" * 50)
    print(f"Total processing time: {total_time:.2f} seconds")
    if test_list:
        print(f"Average time per sample: {total_time / len(test_list):.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline Multimodal Humor Detection"
    )
    add_common_args(parser)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    output_file = args.output or f"baseline_{args.provider}.jsonl"

    check_api_key(args.provider)
    litellm.set_verbose = False

    # Create multimodal agent (full video with audio + text)
    agent = create_agent(
        "multimodal",
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    print(f"Provider: {args.provider} | Model: {agent.model}")

    dataset = load_data(args.dataset)

    run_parallel(agent, dataset, output_file, max_workers=args.max_workers)
