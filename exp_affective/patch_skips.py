"""Patch skipped (pred=None) entries by retrying with text-only fallback."""
import argparse
import json
import os
import time

from openai import OpenAI

from dataset_configs import get_dataset_config

QWEN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = (
    "You are an expert in language and video understanding. "
    "Your task is to analyze the provided text and video, and answer questions about them. "
    "Be concise. Answer with Yes or No first, then give a brief explanation in 2-3 sentences."
)

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


def call_text_only(client, model, query, text, thinking_budget=4096):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"{query}\n The relevant text of the query is: {text}\n"},
    ]
    kwargs = {"model": model, "messages": messages}
    if thinking_budget > 0:
        kwargs["extra_body"] = {"thinking_budget": thinking_budget}

    response = client.chat.completions.create(**kwargs)
    answer = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return answer, usage


def run(args):
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=QWEN_API_BASE)
    config = get_dataset_config("mustard")
    dataset_path = args.dataset or config.get_default_dataset_path()

    from llm_utils import load_data
    dataset = load_data(dataset_path)
    task_query = config.get_task_query()

    # Find skip keys
    skip_keys = []
    seen = set()
    with open(args.input) as f:
        for line in f:
            for k, v in json.loads(line).items():
                if k in seen:
                    continue
                seen.add(k)
                if v['pred'] is None:
                    skip_keys.append(k)

    print(f"Patching {len(skip_keys)} skipped entries from {args.input}")
    print(f"Model: {args.model}, thinking_budget: {args.thinking_budget}")
    print("=" * 60)

    patched = 0
    for i, key in enumerate(skip_keys):
        sample = dataset[key]
        label = config.get_label_field(sample)
        text = config.get_context_field(sample)

        print(f"\n[{i+1}/{len(skip_keys)}] ID={key} label={label}")
        print(f"  Text: {text[:100]}...")

        start = time.time()
        try:
            answer, usage = call_text_only(client, args.model, task_query, text, args.thinking_budget)
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        elapsed = time.time() - start

        pred = parse_prediction(answer)
        is_correct = (pred == label) if pred is not None else False
        patched += 1

        print(f"  Answer: {answer[:120] if answer else 'None'}...")
        print(f"  Pred={pred} Label={label} Correct={is_correct} Time={elapsed:.1f}s")

        result = {
            key: {
                "answer": answer[:500] if answer else None,
                "label": label,
                "pred": pred,
                "correct": is_correct,
                "time": round(elapsed, 1),
                "tokens": usage,
                "has_video": False,
                "text_fallback": True,
            }
        }
        with open(args.input, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nPatched {patched}/{len(skip_keys)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSONL file to patch")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-thinking")
    parser.add_argument("--thinking-budget", type=int, default=4096)
    args = parser.parse_args()
    run(args)
