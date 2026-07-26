#!/bin/bash
# Batch-run the OFFICIAL StableToolBench eval_pass_rate.py over a list of
# converted models. Produces the official solvable pass-rate numbers used in
# the paper's StableToolBench table (GPT-4o / tooleval evaluator).
#
# Prereqs:
#   1) You already ran CTM inference (run_ctm.py) and converted the answers to
#      StableToolBench format, e.g. via:
#         python eval_ctm_toolbench.py --convert_only ...
#      so that ${CONVERTED_PATH}/<model>/<test_set>.json exist.
#   2) An OpenAI key with GPT-4o access (the tooleval evaluator). Set it via env:
#         export OPENAI_API_KEY=sk-...
#      (or drop it in /data/yiningz9/StableToolBench/openai_key.json — the eval
#       harness reads that file too.)
set -e

# --- OpenAI key: read from env, fall back to the StableToolBench key file -----
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f /data/yiningz9/StableToolBench/openai_key.json ]; then
  OPENAI_API_KEY=$(python -c "import json;print(json.load(open('/data/yiningz9/StableToolBench/openai_key.json'))[0]['api_key'])")
fi
export OPENAI_API_KEY
[ -z "${OPENAI_API_KEY:-}" ] && { echo "ERROR: set OPENAI_API_KEY (GPT-4o) for the tooleval evaluator"; exit 1; }

# --- Paths (ctm-rebuttal) -----------------------------------------------------
EVAL_SCRIPT=/data/yiningz9/StableToolBench/toolbench/tooleval/eval_pass_rate.py
CONVERTED_PATH=${CONVERTED_PATH:-/data/yiningz9/ctm-rebuttal/ctm-ai/exp_toolbench/eval_results_official/converted}
SAVE_PATH=${SAVE_PATH:-/data/yiningz9/ctm-rebuttal/ctm-ai/exp_toolbench/eval_results_official/official}
TEST_IDS=/data/yiningz9/StableToolBench/solvable_queries/test_query_ids
EVALUATOR=${EVALUATOR:-tooleval_gpt-4o}
EVAL_TIMES=${EVAL_TIMES:-3}
MAX_THREADS=${MAX_THREADS:-10}

# --- Models & test sets (override via env, space-separated) -------------------
# Default: the Qwen3-8B non-thinking CTM + CoT baseline model tags.
MODELS=(${MODELS:-qwen3_nonthinking_ctm qwen3_nonthinking_cot})
TEST_SETS=(${TEST_SETS:-G2_category G2_instruction G3_instruction})

mkdir -p "$SAVE_PATH"
cd /data/yiningz9/StableToolBench/toolbench/tooleval

total=${#MODELS[@]}
count=0
for model in "${MODELS[@]}"; do
  for test_set in "${TEST_SETS[@]}"; do
    count=$((count + 1))
    if [ ! -f "${CONVERTED_PATH}/${model}/${test_set}.json" ]; then
      echo "[$count] SKIP (no converted file): ${model} ${test_set}"
      continue
    fi
    if [ -f "${SAVE_PATH}/${test_set}_${model}.json" ]; then
      echo "[$count] SKIP (already evaluated): ${model} ${test_set}"
      continue
    fi
    echo "[$count/${total}x${#TEST_SETS[@]}] EVALUATING: ${model} ${test_set}"
    python eval_pass_rate.py \
      --converted_answer_path "${CONVERTED_PATH}" \
      --save_path "${SAVE_PATH}" \
      --reference_model "${model}" \
      --test_ids "${TEST_IDS}" \
      --evaluator "${EVALUATOR}" \
      --evaluate_times "${EVAL_TIMES}" \
      --max_eval_threads "${MAX_THREADS}" \
      --test_set "${test_set}" 2>&1 | grep -E "^(Test set|Solve rate)"
    echo ""
  done
done

echo "=== ALL EVALUATIONS COMPLETE (results in ${SAVE_PATH}) ==="
