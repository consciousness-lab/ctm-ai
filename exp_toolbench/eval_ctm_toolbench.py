"""
Evaluate CTM-AI ToolBench outputs using StableToolBench's evaluation logic.

This script:
1. Converts CTM-AI output ({query_id}_ctm.json) to StableToolBench's converted answer format
2. Runs pass rate evaluation using the same prompts/logic as StableToolBench's eval_pass_rate.py
3. Outputs results compatible with StableToolBench's evaluation pipeline

Usage:
    python eval_ctm_toolbench.py \
        --ctm_output_dir ./results_test/G2_instruction \
        --query_file /path/to/solvable_queries/test_instruction/G2_instruction.json \
        --test_ids_file /path/to/solvable_queries/test_query_ids/G2_instruction.json \
        --save_path ./eval_results \
        --model_name ctm_toolbench \
        --test_set G2_instruction \
        --openai_key YOUR_KEY \
        --evaluate_times 3
"""

import argparse
import json
import os
import sys
import glob
import random
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Enums (mirror StableToolBench's AnswerStatus)
# ---------------------------------------------------------------------------

class AnswerStatus(Enum):
    Unsure = "Unsure"
    Unsolved = "Unsolved"
    Solved = "Solved"


# ---------------------------------------------------------------------------
# Step 1: Convert CTM-AI output to StableToolBench format
# ---------------------------------------------------------------------------

def load_query_metadata(query_file: str) -> Dict[int, Dict]:
    """Load query file and index by query_id for fast lookup."""
    queries = json.load(open(query_file, 'r'))
    index = {}
    for q in queries:
        qid = q['query_id']
        index[qid] = q
    return index


def build_available_tools(api_list: List[Dict], tool_root_dir: Optional[str] = None) -> List[Dict]:
    """Build available_tools list from query's api_list.

    Mimics the format expected by StableToolBench evaluator:
    [{"name": "func_name", "description": "..."}, ...]
    """
    tools = []
    seen = set()
    for api in api_list:
        tool_name = api['tool_name']
        api_name = api['api_name']
        # Create a simple tool entry matching what the evaluator expects
        func_name = f"{api_name}_for_{tool_name}"
        if func_name not in seen:
            tools.append({
                'name': func_name,
                'description': api.get('api_description', ''),
            })
            seen.add(func_name)
    return tools


def convert_ctm_output(ctm_result: Dict, query_meta: Dict) -> Dict:
    """Convert a single CTM-AI output to StableToolBench's converted answer format.

    StableToolBench expects:
    {
        "query": str,
        "available_tools": [...],
        "answer": {
            "method": str,
            "total_steps": int,
            "final_answer": str,
            "answer_details": [nested dict with role/message/next]
        }
    }

    The key requirement: final step must contain a 'Finish' tool call with
    return_type: "give_answer", otherwise the evaluator marks it Unsolved.
    """
    query = ctm_result['query']
    final_answer = ctm_result.get('parsed_answer') or ctm_result.get('final_answer', '')
    available_tools = build_available_tools(query_meta.get('api_list', []))

    # Build answer_details as a nested chain:
    # system -> user -> assistant (CTM reasoning) -> tool (Finish with give_answer)
    finish_args = json.dumps({
        "return_type": "give_answer",
        "final_answer": final_answer
    })

    # Build the chain from leaf to root (then nest)
    # The evaluator walks answer_details[0] -> next[0] -> next[0] -> ...
    # and looks for 'tool' role nodes. The final step must have 'Finish'.

    # Leaf: Finish tool call
    finish_node = {
        "role": "tool",
        "message": {
            "name": "Finish",
            "arguments": finish_args,
            "response": ""
        },
        "next": []
    }

    # Assistant node: CTM's reasoning/answer
    assistant_node = {
        "role": "assistant",
        "message": f"Based on the available tools, I have gathered the following information to answer the query. {final_answer[:500]}",
        "next": [finish_node]
    }

    # User node: the query
    user_node = {
        "role": "user",
        "message": query,
        "next": [assistant_node]
    }

    # System node
    system_node = {
        "role": "system",
        "message": (
            "You are AutoGPT, you can use many tools(functions) to do the following task.\n"
            "Task description: You should use functions to help handle the real time user querys. "
            "Remember to ALWAYS call \"Finish\" function at the end of the task. "
            "And the final answer should contain enough information to show to the user.\n"
            f"Specifically, you have access to the following functions: {json.dumps([t['name'] for t in available_tools])}"
        ),
        "next": [user_node]
    }

    return {
        "query": query,
        "available_tools": available_tools,
        "answer": {
            "method": "CTM",
            "total_steps": 3,  # system + user + assistant + finish
            "final_answer": final_answer,
            "answer_details": [system_node]
        }
    }


def convert_all_ctm_outputs(
    ctm_output_dir: str,
    query_file: str,
) -> Dict[str, Dict]:
    """Convert all CTM-AI output files in a directory to converted answer format.

    Returns: {query_id_str: converted_answer, ...}
    """
    query_index = load_query_metadata(query_file)
    converted = {}

    files = sorted(glob.glob(os.path.join(ctm_output_dir, '*_ctm.json')))
    if not files:
        print(f"Warning: No *_ctm.json files found in {ctm_output_dir}")
        return converted

    for f in files:
        try:
            ctm_result = json.load(open(f, 'r'))
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse {f}, skipping")
            continue

        qid = ctm_result.get('query_id')
        if qid is None:
            # Try to extract from filename
            basename = os.path.basename(f)
            qid = int(basename.split('_')[0])

        query_meta = query_index.get(qid, {})
        if not query_meta:
            print(f"Warning: query_id {qid} not found in query file, skipping")
            continue

        converted_answer = convert_ctm_output(ctm_result, query_meta)
        converted[str(qid)] = converted_answer

    print(f"Converted {len(converted)} CTM-AI outputs")
    return converted


# ---------------------------------------------------------------------------
# Step 2: Evaluate using LLM (same logic as StableToolBench)
# ---------------------------------------------------------------------------

def get_steps(example: Dict) -> Tuple[str, str]:
    """Extract steps from answer_details. Mirrors StableToolBench's utils.get_steps()."""
    answer_details = example["answer"]["answer_details"][0]
    answer_steps = []
    step_cnt = 1
    final_step = ""

    while "next" in answer_details:
        answer_str = answer_details["message"]
        role_str = answer_details["role"]

        if answer_str and role_str == "tool":
            step_text = f"Step {step_cnt}: {answer_str}"
            answer_steps.append(step_text)
            final_step = f"Final step: {answer_str}"
            step_cnt += 1

        if not answer_details["next"]:
            break
        answer_details = answer_details["next"][0]

    return "\n".join(answer_steps), final_step


def check_is_solved_with_llm(
    query: str,
    final_answer: str,
    answer_json: Dict,
    api_key: str,
    model: str = "gemini/gemini-2.5-flash-lite",
) -> AnswerStatus:
    """Evaluate whether the answer solves the query using LLM.

    Uses the same two-stage logic as StableToolBench:
    1. check_answer_status: quick check on final_answer
    2. parse_answer_status: detailed check if stage 1 returns Unsure
    """
    from litellm import completion

    # Stage 1: check_answer_status
    check_prompt = f"""Giving the query and answer, you need give `answer_status` of the answer by following rules:
1. If the answer is a sorry message or not a positive/straight response for the given query, return "Unsolved".
2. If the answer is a positive/straight response for the given query, you have to further check.
2.1 If the answer is not sufficient to determine whether the solve the query or not, return "Unsure".
2.2 If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Solved" or "Unsolved".

Query:
{query}
Answer:
{final_answer}

Respond with ONLY a JSON object: {{"answer_status": "Solved" | "Unsolved" | "Unsure"}}"""

    # Use max_completion_tokens for reasoning models (e.g. gpt-5-mini, o-series)
    is_reasoning_model = any(k in model.lower() for k in ['gpt-5', 'o1', 'o3', 'o4'])
    token_kwarg = {'max_completion_tokens': 1000} if is_reasoning_model else {'max_tokens': 200}
    temp_kwarg = {} if is_reasoning_model else {'temperature': 0.2}

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": check_prompt}],
            **token_kwarg,
            **temp_kwarg,
        )
        text = response.choices[0].message.content.strip()
        status = _parse_answer_status(text)
    except Exception as e:
        print(f"  LLM check_answer_status error: {e}")
        status = AnswerStatus.Unsure

    if status != AnswerStatus.Unsure:
        return status

    # Stage 2: parse_answer_status (detailed check)
    # Truncate answer JSON to avoid token limits
    answer_str = json.dumps(answer_json)
    if len(answer_str) > 3000:
        answer_str = answer_str[:3000] + "..."

    parse_prompt = f"""Giving the query and the correspond execution detail of an answer, you need give `answer_status` of the answer by following rules:
1. If all 'tool' nodes' message indicate that there are errors happened, return "Unsolved"
2. If you find the information in the "final_answer" is not true/valid according to the messages in 'tool' nodes, return "Unsolved"
3. If you are unable to verify the authenticity and validity of the information, return "Unsure"
4. If there are 'tool' node in the chain contains successful func calling and those calling indeed solve the query, return "Solved"

Query:
{query}
Answer:
{answer_str}

Respond with ONLY a JSON object: {{"answer_status": "Solved" | "Unsolved" | "Unsure"}}"""

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": parse_prompt}],
            **token_kwarg,
            **temp_kwarg,
        )
        text = response.choices[0].message.content.strip()
        status = _parse_answer_status(text)
    except Exception as e:
        print(f"  LLM parse_answer_status error: {e}")
        status = AnswerStatus.Unsure

    return status


def _parse_answer_status(text: str) -> AnswerStatus:
    """Parse LLM output to extract AnswerStatus."""
    text_lower = text.lower()
    # Try to parse JSON
    try:
        # Handle markdown code blocks
        if '```' in text:
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        obj = json.loads(text.strip())
        status_str = obj.get('answer_status', 'Unsure')
        return AnswerStatus(status_str)
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Fallback: keyword matching
    if '"solved"' in text_lower or "'solved'" in text_lower:
        if '"unsolved"' in text_lower or "'unsolved'" in text_lower:
            return AnswerStatus.Unsure
        return AnswerStatus.Solved
    if '"unsolved"' in text_lower or "'unsolved'" in text_lower:
        return AnswerStatus.Unsolved
    return AnswerStatus.Unsure


# ---------------------------------------------------------------------------
# Step 3: Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_pass_rate(
    converted_answers: Dict[str, Dict],
    test_ids: List[str],
    api_key: str,
    model: str = "gemini/gemini-2.5-flash-lite",
    evaluate_times: int = 3,
    max_eval_threads: int = 3,
    save_path: Optional[str] = None,
    model_name: str = "ctm_toolbench",
    test_set: str = "G2_instruction",
) -> Dict:
    """Run pass rate evaluation on converted answers.

    Returns label_cnt dict compatible with StableToolBench's output format.
    """
    label_cnt = {}

    # Filter to only test_ids
    eval_ids = [qid for qid in converted_answers if qid in test_ids]
    print(f"Evaluating {len(eval_ids)} queries (out of {len(converted_answers)} converted, {len(test_ids)} in test set)")

    if not eval_ids:
        print("Warning: No matching query IDs found between converted answers and test_ids")
        # Evaluate all converted answers instead
        eval_ids = list(converted_answers.keys())
        print(f"Falling back to evaluating all {len(eval_ids)} converted answers")

    def compute_single(query_id: str, evaluate_time: int):
        example = converted_answers[query_id]
        answer_steps, final_step = get_steps(example)

        # Key check from StableToolBench: final step must contain 'Finish'
        if "'name': 'Finish'" not in final_step:
            return query_id, AnswerStatus.Unsolved, evaluate_time

        # Empty or give_up check
        final_answer = example['answer'].get('final_answer', '')
        if not final_answer or 'give_up_and_restart' in final_answer:
            return query_id, AnswerStatus.Unsolved, evaluate_time

        # LLM-based evaluation
        status = check_is_solved_with_llm(
            query=example['query'],
            final_answer=final_answer,
            answer_json=example['answer'],
            api_key=api_key,
            model=model,
        )
        return query_id, status, evaluate_time

    # Run evaluations
    futures = []
    with ThreadPoolExecutor(max_workers=max_eval_threads) as pool:
        for qid in eval_ids:
            for t in range(evaluate_times):
                futures.append(pool.submit(compute_single, qid, t))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", ncols=100):
            try:
                query_id, is_solved, evaluate_time = future.result()
            except Exception as e:
                print(f"  Evaluation error: {e}")
                continue

            example = converted_answers[query_id]
            answer_steps, final_step = get_steps(example)
            tool_names = [t.get('name', '') for t in example.get('available_tools', [])]

            if query_id not in label_cnt:
                label_cnt[query_id] = {}
            label_cnt[query_id]["query"] = example["query"]
            label_cnt[query_id]["tool_names"] = tool_names
            label_cnt[query_id]["answer_steps"] = answer_steps
            label_cnt[query_id]["final_step"] = final_step
            if "is_solved" not in label_cnt[query_id]:
                label_cnt[query_id]["is_solved"] = {}
            label_cnt[query_id]["is_solved"][evaluate_time] = str(is_solved)

            # Save progressively
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                json.dump(
                    label_cnt,
                    open(os.path.join(save_path, f"{test_set}_{model_name}.json"), "w"),
                    ensure_ascii=False, indent=2
                )

    # Compute final scores
    scores = []
    for runtime in range(evaluate_times):
        score = 0
        for query_id in label_cnt:
            solved_dict = label_cnt[query_id].get('is_solved', {})
            solved_dict = {int(k): v for k, v in solved_dict.items()}
            if runtime not in solved_dict:
                continue
            if solved_dict[runtime] == "AnswerStatus.Solved":
                score += 1
            elif solved_dict[runtime] == "AnswerStatus.Unsure":
                score += 0.5
        scores.append(score / max(len(label_cnt), 1))

    solve_rate = np.mean(scores) * 100 if scores else 0
    std_dev = np.std(scores).item() * 100 if scores else 0

    print(f"\n{'='*60}")
    print(f"Test set: {test_set}. Model: {model_name}")
    print(f"Queries evaluated: {len(label_cnt)}")
    print(f"Solve rate: {solve_rate:.1f}% (std: {std_dev:.1f}%)")
    print(f"{'='*60}")

    # Per-query breakdown
    for qid in sorted(label_cnt.keys(), key=lambda x: int(x)):
        solved_dict = label_cnt[qid].get('is_solved', {})
        statuses = list(solved_dict.values())
        print(f"  qid={qid}: {statuses}")

    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        result_json_path = os.path.join(save_path, f"{test_set}_{model_name}.json")
        json.dump(label_cnt, open(result_json_path, "w"), ensure_ascii=False, indent=2)

        # Also save converted answers (for potential preference evaluation later)
        converted_dir = os.path.join(save_path, "converted", model_name)
        os.makedirs(converted_dir, exist_ok=True)
        converted_path = os.path.join(converted_dir, f"{test_set}.json")
        json.dump(converted_answers, open(converted_path, "w"), ensure_ascii=False, indent=2)

        # CSV output
        csv_path = os.path.join(save_path, f"{test_set}_{model_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["query", "available_tools", "model_intermediate_steps",
                             "model_final_step", "model", "query_id", "is_solved"])
            for qid in label_cnt:
                writer.writerow([
                    label_cnt[qid]["query"],
                    label_cnt[qid]["tool_names"],
                    label_cnt[qid]["answer_steps"],
                    label_cnt[qid]["final_step"],
                    model_name,
                    qid,
                    label_cnt[qid]["is_solved"],
                ])

        print(f"\nResults saved to: {save_path}")
        print(f"  JSON: {result_json_path}")
        print(f"  CSV:  {csv_path}")
        print(f"  Converted answers: {converted_path}")

    return label_cnt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CTM-AI ToolBench outputs")
    parser.add_argument('--ctm_output_dir', type=str, required=True,
                        help='Directory containing *_ctm.json output files')
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to query JSON (e.g., solvable_queries/test_instruction/G2_instruction.json)')
    parser.add_argument('--test_ids_file', type=str, default=None,
                        help='Path to test IDs JSON (e.g., solvable_queries/test_query_ids/G2_instruction.json). '
                             'If not provided, evaluates all queries in ctm_output_dir.')
    parser.add_argument('--save_path', type=str, default='./eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, default='ctm_toolbench',
                        help='Model name for result files')
    parser.add_argument('--test_set', type=str, default='G2_instruction',
                        help='Test set name')
    parser.add_argument('--eval_model', type=str, default='gemini/gemini-2.5-flash-lite',
                        help='LLM model to use for evaluation')
    parser.add_argument('--evaluate_times', type=int, default=3,
                        help='Number of times to evaluate each query')
    parser.add_argument('--max_eval_threads', type=int, default=3,
                        help='Max parallel evaluation threads')
    # For converting only (no eval)
    parser.add_argument('--convert_only', action='store_true',
                        help='Only convert CTM outputs, skip LLM evaluation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Step 1: Convert
    print("Step 1: Converting CTM-AI outputs...")
    converted = convert_all_ctm_outputs(
        ctm_output_dir=args.ctm_output_dir,
        query_file=args.query_file,
    )

    if not converted:
        print("No outputs to evaluate. Exiting.")
        sys.exit(1)

    if args.convert_only:
        # Save converted answers and exit
        os.makedirs(args.save_path, exist_ok=True)
        converted_dir = os.path.join(args.save_path, "converted", args.model_name)
        os.makedirs(converted_dir, exist_ok=True)
        out_path = os.path.join(converted_dir, f"{args.test_set}.json")
        json.dump(converted, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"Converted answers saved to: {out_path}")
        sys.exit(0)

    # Load test IDs
    if args.test_ids_file and os.path.exists(args.test_ids_file):
        test_ids = list(json.load(open(args.test_ids_file, 'r')).keys())
        print(f"Loaded {len(test_ids)} test IDs from {args.test_ids_file}")
    else:
        test_ids = list(converted.keys())
        print(f"No test_ids_file provided, using all {len(test_ids)} converted query IDs")

    # Step 2: Evaluate
    print(f"\nStep 2: Evaluating pass rate (model={args.eval_model}, times={args.evaluate_times})...")
    evaluate_pass_rate(
        converted_answers=converted,
        test_ids=test_ids,
        api_key=os.environ.get('GEMINI_API_KEY', os.environ.get('OPENAI_API_KEY', '')),
        model=args.eval_model,
        evaluate_times=args.evaluate_times,
        max_eval_threads=args.max_eval_threads,
        save_path=args.save_path,
        model_name=args.model_name,
        test_set=args.test_set,
    )
