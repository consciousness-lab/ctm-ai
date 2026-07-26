# CTM-AI StableToolBench Experiments & Evaluation

Run and evaluate CTM-AI (and CoT baselines) on **StableToolBench** (solvable
pass rate, MirrorAPI-Cache server). Supports two backbones:

- **Gemini** (`--ctm_name tooluse_ctm`, default) — cloud, needs `GEMINI_API_KEY`.
- **Local Qwen3-8B, non-thinking** (`--ctm_name tooluse_ctm_qwen3`) — self-hosted
  via vLLM, no cloud key needed for the base model. This is the paper's
  "Qwen3-8B CoT + CTM-AI" backbone.

## Files

```
exp_toolbench/
├── run_ctm.py                    # CTM-AI inference entry point
├── run_toolbench.sh              # Inference example script (gemini)
├── eval_ctm_toolbench.py         # Local eval (convert + gemini pass-rate approx.)
├── convert_cot_to_ctm_format.py  # Convert StableToolBench CoT outputs -> CTM format
├── run_all_official_eval.sh      # Batch OFFICIAL eval_pass_rate.py (GPT-4o tooleval)
└── README.md
```

## Hosting prerequisites (three services)

The StableToolBench tool server + the local base model must be up. See the
project memory / `../../program-toolbench.md` for full details.

| Service | GPU | Port | Command |
|---|---|---|---|
| Qwen3-8B (base model, non-thinking) | any | 8001 | `CUDA_VISIBLE_DEVICES=8 vllm serve /data/models/Qwen3-8B --port 8001 --served-model-name Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.9 --max-model-len 40960` |
| MirrorAPI simulator (`mirrorapi`) | any | 60010 | `CUDA_VISIBLE_DEVICES=9 vllm serve /data/yiningz9/StableToolBench/MirrorAPI-Cache --port 60010 --served-model-name mirrorapi --gpu-memory-utilization 0.9 --max-model-len 32768` |
| MirrorAPI-Cache FastAPI (`/virtual`) | CPU | 8080 | from `StableToolBench/server/`: `python main_mirrorapi_cache.py` |

Health check (end-to-end tool server): `python /data/yiningz9/StableToolBench/test_server.py`
→ expects `{"error":"","response":"..."}`.

## Environment

```bash
conda activate ctm-space
export PYTHONPATH=/data/yiningz9/ctm-rebuttal/ctm-ai
export SERVICE_URL="http://localhost:8080/virtual"
export VLLM_API_BASE="http://localhost:8001/v1"   # local Qwen3 endpoint
export OPENAI_API_KEY=fake                          # litellm needs it to exist
# For the gemini backbone only:
# export GEMINI_API_KEY="your-gemini-api-key"
```

**Non-thinking toggle:** `vllm/` models default to non-thinking
(`enable_thinking=false`, baked into `ctm_ai/utils/litellm_utils.py`). Set
`export VLLM_THINKING=1` to run the thinking variant instead.

## Step 1 — Inference (local Qwen3-8B non-thinking)

Output dirs must be created first (`os.mkdir` is not recursive).

Single query (debug):
```bash
cd /data/yiningz9/ctm-rebuttal/ctm-ai/exp_toolbench
mkdir -p ./results_qwen3_nonthinking/G2_instruction
python run_ctm.py \
  --tool_root_dir /data/yiningz9/StableToolBench/toolenv2404_filtered \
  --openai_key fake --max_observation_length 1024 --method ctm \
  --input_query_file /data/yiningz9/StableToolBench/solvable_queries/test_instruction/G2_instruction.json \
  --output_answer_file ./results_qwen3_nonthinking/G2_instruction \
  --toolbench_key "" --ctm_name tooluse_ctm_qwen3 --query_id 4746
```

All three test sets, parallel:
```bash
for group in G2_category G2_instruction G3_instruction; do
  mkdir -p ./results_qwen3_nonthinking/${group}
  python run_ctm.py \
    --tool_root_dir /data/yiningz9/StableToolBench/toolenv2404_filtered \
    --openai_key fake --max_observation_length 1024 --method ctm \
    --input_query_file /data/yiningz9/StableToolBench/solvable_queries/test_instruction/${group}.json \
    --output_answer_file ./results_qwen3_nonthinking/${group} \
    --toolbench_key "" --ctm_name tooluse_ctm_qwen3 --num_processes 4
done
```

Each query writes `{query_id}_ctm.json`:
`{query, query_id, final_answer, weight_score, parsed_answer}` plus a per-query
`ctm_iterations_{query_id}.jsonl` trajectory log.

## Step 2 — Evaluation

### 2a. Quick local pass-rate (gemini judge, approximate)
```bash
python eval_ctm_toolbench.py \
  --ctm_output_dir ./results_qwen3_nonthinking/G2_instruction \
  --query_file /data/yiningz9/StableToolBench/solvable_queries/test_instruction/G2_instruction.json \
  --test_ids_file /data/yiningz9/StableToolBench/solvable_queries/test_query_ids/G2_instruction.json \
  --save_path ./eval_results --test_set G2_instruction --evaluate_times 3
```

### 2b. OFFICIAL pass-rate (GPT-4o tooleval — the paper numbers)
Step 1: convert CTM outputs to StableToolBench format (per test set, under a model tag):
```bash
for group in G2_category G2_instruction G3_instruction; do
  python eval_ctm_toolbench.py \
    --ctm_output_dir ./results_qwen3_nonthinking/${group} \
    --query_file /data/yiningz9/StableToolBench/solvable_queries/test_instruction/${group}.json \
    --save_path ./eval_results_official --model_name qwen3_nonthinking_ctm \
    --test_set ${group} --convert_only
done
```
This writes `eval_results_official/converted/qwen3_nonthinking_ctm/<group>.json`.

Step 2: run the official evaluator over all converted models:
```bash
export OPENAI_API_KEY=sk-...            # GPT-4o key for tooleval
MODELS="qwen3_nonthinking_ctm" TEST_SETS="G2_category G2_instruction G3_instruction" \
  bash run_all_official_eval.sh
# results -> eval_results_official/official/<test_set>_<model>.json
```

### CoT baseline conversion (for comparison rows)
StableToolBench's own CoT runs (e.g. `StableToolBench/run_qwen3_cot.py`) emit
`{query_id}_CoT@1.json`. Convert them into CTM format so the same evaluator
applies:
```bash
python convert_cot_to_ctm_format.py \
  --cot_dir /data/yiningz9/StableToolBench/data/answer/qwen3_8b_nonthinking/G2_instruction \
  --query_file /data/yiningz9/StableToolBench/solvable_queries/test_instruction/G2_instruction.json \
  --output_dir ./results_qwen3_nonthinking_cot/G2_instruction
# then eval_ctm_toolbench.py --convert_only ... --model_name qwen3_nonthinking_cot
```

## Test sets

`G2_category` (I2-Category), `G2_instruction` (I2-Instruction),
`G3_instruction` (I3-Instruction). Query files live under
`StableToolBench/solvable_queries/test_instruction/`, official IDs under
`.../test_query_ids/`. Queries whose tools are absent from
`toolenv2404_filtered` are auto-skipped.

## Metrics collected

- **Pass Rate** (Solved=1, Unsure=0.5, Unsolved=0), averaged over `evaluate_times`.
- Per-run token usage / model-call counts: `CTM.get_usage_stats()` (processors)
  + `get_parse_usage_stats()` (parse). Tool calls = `/virtual` hits.
- Latency: wall-clock per query.
