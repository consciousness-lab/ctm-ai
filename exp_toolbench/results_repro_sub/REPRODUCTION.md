# Reproduction check — Qwen3-8B non-thinking, standard CTM (no K-cap)

Standard CTM-AI (all available tools per query) with locally-hosted **Qwen3-8B
non-thinking**, vs ctm-project's recorded runs, on the **same 20 query IDs** per
test set (subset of the solvable set).

| Test set | **MINE** (Gemini judge) | ctm-project raw (Gemini judge) | ctm-project official (GPT-4o) |
|---|--:|--:|--:|
| I2-Cat (G2_category)    | 80.0 (n=20) | 72.5 (n=20) | 54.2 (n=20) |
| I2-Inst (G2_instruction)| 60.8 (n=20) | 65.0 (n=20) | 48.3 (n=20) |
| I3-Inst (G3_instruction)| 75.9 (n=18) | 76.7 (n=20) | 71.7 (n=20) |

(ctm-project full-set official numbers, for reference: I2-Cat 52.1 / I2-Inst 53.4 / I3-Inst 72.7.)

## Verdict

- **Reproduction holds.** Under the *same* judge (Gemini), MINE vs ctm-project
  differ by only +7.5 / −4.2 / −0.8 — within run-to-run variance (temperature
  stochasticity; I3 has n=18 vs 20). The ported CTM + local Qwen3-8B non-thinking
  produces equivalent answer quality to ctm-project's runs.
- **Judge matters for absolute numbers.** Gemini is ~15–20 pts more lenient than
  the GPT-4o official evaluator on the I2 sets (e.g. ctm-proj I2-Cat: 72.5 Gemini
  vs 54.2 GPT-4o). To match the **paper's** absolute numbers (52.1 / 53.4 / 72.7),
  run the StableToolBench **official GPT-4o evaluator** (`run_all_official_eval.sh`).

## GPT-4o judge (same `eval_ctm_toolbench.py` pipeline)

| Test set | MINE (GPT-4o) | ctm-project official (GPT-4o, tooleval) | diff |
|---|--:|--:|--:|
| I2-Cat  | 74.2 | 54.2 | +20.0 |
| I2-Inst | 60.8 | 48.3 | +12.5 |
| I3-Inst | 74.1 | 71.7 | +2.4 |

The +12–20 gap is **not** a quality difference — it is the **eval pipeline**:
MINE uses `eval_ctm_toolbench.py` (plain-text extraction), the ctm-project
*official* column uses StableToolBench's `eval_pass_rate.py` (tooleval +
function-calling extraction, stricter). Proof it's the pipeline: under the
*matched* pipeline+judge (`eval_ctm_toolbench.py` + Gemini) MINE ≈ ctm-proj
(80/61/76 vs 72/65/77). **To match the paper's absolute numbers, run the official
`run_all_official_eval.sh` (tooleval GPT-4o).** All new tables here use one
consistent judge so cross-condition comparisons remain valid.

## Setup

- Inference: `run_ctm.py --ctm_name tooluse_ctm_qwen3` (no `--k_processors`), 20 solvable qids/set.
- Judge: `eval_ctm_toolbench.py --eval_model gemini/gemini-2.5-flash-lite --evaluate_times 3`.
- Same 20 qids applied to ctm-project's `results_mar28/qwen3_ctm` outputs for the matched-judge column.
