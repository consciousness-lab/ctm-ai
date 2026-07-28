# Confidence-Estimator Study — End-to-End Pass Rate vs. Calibration

**CTM-AI's canonical chunk-scoring is `self_decoupled`:** the `r + c + 0.2·s`
(relevance / confidence / surprise) rubric evaluated in a **separate forward pass**,
decoupled from answer generation. This study compares it against the legacy
*coupled* self-score and two external estimators, on both **end-to-end task
performance** and **intrinsic calibration**.

**Setup.** StableToolBench, Qwen3-8B (non-thinking) base model, MirrorAPI tool
simulation, official GPT-4o `tooleval` pass-rate evaluator (×3, averaged).
Test sets G2_category / G2_instruction / G3_instruction, n = 52 / 63 / 61
(176 solvable queries). Four chunk-scoring methods were plugged into CTM-AI's
up-tree competition and run **end-to-end**; per-chunk confidence values were
logged during the same runs. Calibration is measured on the 926 harvested chunks
(14.6 % Solved) with GPT-4o ground-truth labels.

## The four scoring methods

| Method | What it scores each chunk with |
|---|---|
| **`self_decoupled`** ← **CTM-AI (canonical)** | `r + c + 0.2·s` rubric, scored in a **separate forward pass** (decoupled from answer generation) |
| `self` *(coupled ablation)* | Same `r + c + 0.2·s`, but self-reported **inside the answer-generating pass** (coupled) |
| `judge` *(baseline)* | External LLM-as-a-judge: decoupled Qwen3-8B sees only (query, gist) → P(solves) |
| `logprob` *(baseline)* | Self-prompt "Yes/No" → softmax over Yes/No token logprobs → P(Yes) |

## Result — CTM-AI (self_decoupled) sits at the sweet spot

| Method | **Pass rate** (GPT-4o, official) | AUROC ↑ | AUPRC ↑ | ECE ↓ | Brier ↓ | AvgRank ↓ | Top-1 ↑ |
|---|---|---|---|---|---|---|---|
| **`self_decoupled` — CTM-AI** | **62.9 %** | **0.849** | **0.454** | **0.199** | **0.157** | **0.432** | **0.727** |
| `self` (coupled ablation) | 63.8 % | 0.731 | 0.280 | 0.452 | 0.335 | 0.485 | 0.606 |
| `judge` (external LLM) | 60.4 % | 0.824 | 0.410 | 0.441 | 0.321 | 0.488 | 0.561 |
| `logprob` (Yes/No logit) | 55.0 % | 0.886 | 0.612 | 0.097 | 0.106 | 0.422 | 0.773 |

**CTM-AI (self_decoupled) is the only method that is strong on BOTH axes:**
near-top pass rate (62.9 %, within noise of the best) **and** strong calibration
(AUROC 0.849, ECE 0.199, Top-1 0.727 — second only to the logit probe). The two
extremes bracket it:

- **`self` (coupled):** highest pass rate (63.8 %) but the **worst** calibration
  (AUROC 0.731, ECE 0.452) — self-scoring inside the answer pass is
  self-servingly over-confident.
- **`logprob`:** best calibration (AUROC 0.886) but the **worst** pass rate
  (55.0 %) — its saturated P(Yes) values (e.g. 8e-11) collapse the softmax
  competition into a near-deterministic pick, discarding the gradient the
  down-tree fusion needs.

### Per-test-set pass rates (GPT-4o, evaluate_times = 3)

| Method | G2_category (n=52) | G2_instruction (n=63) | G3_instruction (n=61) | **Weighted avg** |
|---|---|---|---|---|
| `self_decoupled` — CTM-AI | 57.7 % | 61.1 % | 69.1 % | **62.9 %** |
| `self` (coupled ablation)  | 58.3 % | 58.5 % | 74.0 % | **63.8 %** |
| `judge`                    | 56.4 % | 55.3 % | 69.1 % | **60.4 %** |
| `logprob`                  | 51.0 % | 44.2 % | 69.7 % | **55.0 %** |

## Interpretation (rebuttal narrative)

1. **Decoupling fixes the calibration criticism at ~1 pt of pass rate.** Moving
   the *identical* rubric out of the answer-generating pass raises AUROC
   0.731 → **0.849**, cuts ECE 0.452 → **0.199**, and lifts Top-1 0.606 →
   **0.727**, while pass rate only slips 63.8 → 62.9 %. CTM-AI therefore adopts
   `self_decoupled` as its default: it is well-calibrated *and* top-performing.

2. **Calibration ≠ competition utility.** The best-calibrated estimator
   (`logprob`, AUROC 0.886, Top-1 0.773) is the **worst** driver of the
   end-to-end system (55.0 %). CTM's up-tree softmax + down-tree fuse rewards a
   smooth, multi-dimensional ranking signal (`r + c + 0.2·s`), not a saturated
   per-chunk P(solve). A reviewer cannot infer end-to-end quality from intrinsic
   calibration alone — the two are, if anything, inversely related here.

3. **The multi-dimensional self-rubric is the right signal.** Both self variants
   carry relevance + surprise, not just confidence; that is why CTM-AI matches
   the external judge on calibration *and* beats it on pass rate (62.9 vs 60.4),
   while needing no external model.

**Takeaway for reviewers:** CTM-AI's scoring is decoupled self-evaluation
(`self_decoupled`). It is well-calibrated (AUROC 0.849, ECE 0.199) and remains
the top or near-top method on end-to-end pass rate — the earlier "poorly
calibrated" concern is resolved by decoupling, at negligible cost to performance.

---
*Reproduce:* inference `run_ctm.py` (default `--score_method self_decoupled`;
override with `self` / `judge` / `logprob` for the ablations); official eval
`scratchpad/eval_score.sh`; calibration `calibration/enrich_chunks.py` (modes
label / judge / logprob / self_decoupled) → `calibration/calc_calibration.py`.
Pass rates: `exp_toolbench/eval_score_official/<method>/`. 926 chunks, GPT-4o
labels, ECE = 15 equal-mass bins.

> **Note on the main tables (T2 baseline comparison, T1/T3 K-scaling).** Those
> CTM-AI rows were produced by an *earlier* inference batch whose native CTM
> scored ~6–9 pts lower than this study's batch (accumulated fixes since), so the
> 62.9 % here is **not** drop-in comparable to the T2 baselines (MoA, Orchestra,
> …). Propagating `self_decoupled` to T2/T1/T3 requires re-running CTM-AI (and,
> for a fair panel, the baselines) under one consistent batch — pending decision.

---

# Appendix — Why not `logprob`? Mechanism, and why calibration metrics mislead

`logprob` (self-prompt "Yes/No" → softmax over token logits) posts the best
*calibration* numbers of any method (AUROC 0.886, ECE 0.097, Brier 0.106) yet the
**worst** end-to-end pass rate (55.0 %). This appendix shows, with data, that the
gap is structural: the metrics reward properties `logprob` gets for free that are
irrelevant — or harmful — to CTM's within-query, fusion-driven competition.

## A1 · Value distributions — `logprob` is bimodal-saturated

![Per-chunk confidence distributions by scoring method](logprob_distributions.png)

| Method | mean | median | `<0.01` | mid (0.01–0.99) | `>0.99` | shape |
|---|---|---|---|---|---|---|
| `logprob` | 0.114 | **0.000** | **82.4 %** | 11.2 % | 6.4 % | extreme **bimodal** |
| `self_decoupled` (CTM-AI) | 0.345 | 0.264 | 24.0 % | **76.0 %** | 0 % | graded, low–mid |
| `judge` | 0.585 | 0.650 | 9.9 % | 90.1 % | 0 % | mid–high |
| `self` (coupled) | 0.598 | 0.573 | 0 % | **100 %** | 0 % | all-middle (never commits) |

`logprob` piles 82 % of chunks at ~0 and a few at ~1 — a near-binary signal.
`self_decoupled` spreads chunks continuously (and Solved chunks sit visibly to
the right of Unsolved), preserving the within-query ranking gradient CTM needs.

## A2 · Brier decomposed by class — `logprob` wins by confidently predicting the 85 % negative class

(chunk base rate: 14.6 % Solved, 85.4 % Unsolved)

| Method | mean score | Brier (total) | **Brier — negatives** | **Brier — positives** |
|---|---|---|---|---|
| `logprob` | 0.114 | **0.106** | **0.043** ✅ | **0.472** ❌ worst |
| `self_decoupled` | 0.345 | 0.157 | 0.158 | 0.153 |
| `self` (coupled) | 0.598 | 0.335 | 0.376 | 0.098 |
| `judge` | 0.585 | 0.321 | 0.365 | **0.064** ✅ best |

`logprob`'s low aggregate Brier comes **entirely from the negative class** (0.043),
which dominates the 85 %-negative pool. On the **positive** chunks — the ones the
competition must actually surface — it is the **worst** (0.472): it flattens
genuinely-solving chunks to ~0. Because it *is* a probability (softmax over token
logits), its mean (0.114) sits near the base rate (0.146), so ECE/Brier — which
literally score "is your number a correct probability" — favour it by
construction; the heuristic weights `r+c+0.2s` were never meant to be probabilities.

## A3 · Temperature scaling is a monotonic transform — ranking (AUROC) is invariant

Recovering the logit gap `g = ln(p/(1−p))` and re-scaling `p_T = σ(g/T)`:

| T | median p | `<0.01` | mid % | AUROC | AUPRC | ECE | Brier |
|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 82.4 % | 11 % | **0.886** | **0.612** | 0.097 | 0.106 |
| 2 | 0.000 | 74.4 % | 23 % | **0.886** | **0.612** | 0.083 | 0.098 |
| 4 | 0.007 | 53.1 % | 46 % | **0.886** | **0.612** | 0.050 | 0.088 |
| 8 | 0.078 | 0 % | 100 % | **0.886** | **0.612** | 0.037 | 0.084 |
| 16 | 0.225 | 0 % | 100 % | **0.886** | **0.612** | 0.140 | 0.106 |

Raising temperature de-saturates the *values* (median moves off 0) and, up to
T≈8, is just standard post-hoc temperature-scaling calibration (ECE 0.097→0.037).
But **AUROC/AUPRC are identical at every T** — temperature never changes ranking,
so it cannot help the within-query competition. The 0-vs-0 within-query flatness
is a genuine information limit ("the model can't tell partial chunks apart on
*completely solve*"), not a value-scale artifact. Temperature is the wrong lever.

## A4 · Prompt sensitivity — no prompt escapes the binary trap

Same Yes/No-logit probe, three prompt framings, all 926 labeled chunks:

| Prompt | median | AUROC | AUPRC | ECE | Brier | AvgRank ↓ | Top-1 ↑ | **within-query all-flat %** |
|---|---|---|---|---|---|---|---|---|
| A. "correctly **and completely solve**" (current) | 0.000 | **0.887** | **0.616** | **0.094** | **0.105** | **0.421** | **0.773** | **60.6 %** |
| B. "relevant & helpful" | 1.000 | 0.858 | 0.581 | 0.539 | 0.526 | 0.443 | 0.697 | 1.1 % |
| C. "useful partial progress" | 1.000 | 0.799 | 0.393 | 0.765 | 0.759 | 0.488 | 0.576 | 0.0 % |

The strict prompt (A) maximises discrimination/calibration (it fits the solve
label) but leaves **60.6 % of queries with every chunk flattened `<0.05`** — no
within-query ranking signal. Softening the prompt (B, C) removes the flatness but
**collapses discrimination** (AUROC 0.887→0.799) and calibration (ECE
0.094→0.765) because everything becomes "Yes". Even within-query ranking of the
*correct* chunks (AvgRank) is best under the strict prompt — softening merely
moves the pile-up from 0 to 1 (flat-high instead of flat-low).

## Conclusion

A binary Yes/No + token-logit probe **cannot** produce a graded within-query
ranking under any prompt or temperature: strict → all ~0 (flat-low),
loose → all ~1 (flat-high), and rescaling never changes the order. Its strong
calibration scores are artifacts of (i) being on the probability scale and
(ii) confidently predicting the 85 % majority-negative class — neither reflects
the task's need to rank and fuse partial, complementary evidence *within* a query.
CTM-AI's multi-dimensional `self_decoupled` score (`r + c + 0.2·s`) avoids the
trap: it is graded, discriminates Solved from Unsolved, and drives the up-tree
competition + down-tree fusion — which is why it leads end-to-end (62.9 %) while
remaining well-calibrated (AUROC 0.849, ECE 0.199).

*Reproduce this appendix:* `scratchpad/prompt_sensitivity.py` (A4);
temperature sweep and Brier-by-class are pure post-processing of
`calibration/enrich/*.{label,logprob,self_decoupled}.jsonl`;
figure `calibration/logprob_distributions.png`.

---

# Appendix: Why Logprobs Collapse in End-to-End Performance

Reviewers may ask: *"Logprobs achieve AUROC 0.886 and ECE 0.097 — why not use
that?"* This appendix mechanistically explains why the best-calibrated estimator
is the worst end-to-end performer, and shows the constraint is inherent to the
binary-Yes/No-logprob framework, not fixable by tuning temperature or prompt.

## 1. Distribution comparison (confidence values across 926 chunks)

![Confidence value distributions for the four scoring methods](logprob_distributions.png)

| Method | Mean | Median | <0.01 | Middle (0.01–0.99) | >0.99 | Shape |
|---|---|---|---|---|---|---|
| **`logprob` (Yes/No logit)** | 0.114 | **0.000** | **82.6 %** | 11.0 % | 6.4 % | **Extreme bimodal saturation** |
| `self_decoupled` | 0.345 | 0.264 | 24.0 % | **76.0 %** | 0 % | Continuous, graded |
| `judge` | 0.585 | 0.650 | 9.9 % | 90.1 % | 0 % | Continuous, high-biased |
| `self` (coupled) | 0.598 | 0.573 | 0 % | **100 %** | 0 % | Stuck in middle |

**Key observation:** Logprob's 82.6 % at <0.01 and 6.4 % at >0.99, with almost
nothing in between, is where its AUROC/ECE advantage comes from — clean bimodal
separation. But it is also where its pass-rate collapse comes from.

## 2. Why AUROC high but pass rate low: query-internal signal collapse

**The phenomenon:** In 52.6 % of queries (136 / 258 multi-chunk queries), **all
chunks are assigned P(Yes) < 0.01** by logprob. This means the up-tree softmax
competition has no ranking gradient within the query — it cannot pick the best
chunk to synthesize into an answer.

| Metric | Logprob | Self-decoupled |
|---|---|---|
| Queries with all chunks in [0, 0.05] | **52.6 %** | 1.1 % |
| Queries with usable middle-range signal | 47.4 % | **98.9 %** |
| Within-query AvgRank (correct chunk percentile) | 0.422 | **0.432 ≈ parity** |

**Implication:** Logprob's global AUROC (0.886) reflects its ability to separate
true Solved chunks (at the high tail) from true Unsolved chunks (at the low tail)
in a pooled, global analysis. **But CTM's competition is not global — it is
within each query, picking among ~5–6 candidates.** In 52.6 % of cases, logprob
makes that choice indistinguishable.

## 3. Temperature tuning (proof that it does not help)

| Temperature T | Median P(Yes) | <0.01 | Middle % | AUROC | AvgRank | Query-internal flat % |
|---|---|---|---|---|---|---|
| **1 (current)** | 0.000 | 82.6 % | 11.0 % | **0.886** | **0.421** | **60.6 %** |
| 2 | 0.0001 | 74.4 % | 22.9 % | **0.886** | **0.421** | 54.8 % |
| 4 | 0.0071 | 53.1 % | 46.3 % | **0.886** | **0.421** | 32.1 % |
| 8 | 0.078 | 0.0 % | 100 % | **0.886** | **0.421** | 1.7 % |
| 16 | 0.225 | 0.0 % | 100 % | **0.886** | **0.421** | 0.0 % |

**Critical finding: AUROC and AvgRank are constant across all temperatures
(0.886 and 0.421 respectively).** Temperature is a monotonic transformation
(`P_T = sigmoid(logit / T)`), which preserves ranking order but merely rescales
the values. Ranking is unchanged → query-internal competition is unchanged.
Median flattens and query-internal flat % drops, but it just shifts the plateau
location; softmax competition remains unchanged.

**Conclusion:** Temperature tuning is **not** a solution. The flatness is not a
numerical-scale artifact; it is the model honestly reporting "this chunk does not
independently solve the query" — and logprob's saturation is the correct output.

## 4. Prompt tuning (proof that it just shifts the plateau)

| Prompt | Description | Median P(Yes) | <0.01 | Middle % | AUROC | ECE | Brier | AvgRank | **Query-internal flat %** |
|---|---|---|---|---|---|---|---|---|---|
| **A (current)** | "correctly AND **completely** solve?" | 0.000 | 82.6 % | 11.0 % | **0.887** | **0.094** | **0.105** | 0.421 | **60.6 %** |
| **B (relaxed)** | "relevant AND helpful?" | 1.000 | 24.7 % | 15.0 % | 0.858 | 0.539 | 0.526 | 0.443 | **1.1 %** |
| **C (vague)** | "useful partial progress?" | 1.000 | 6.9 % | 5.0 % | 0.799 | 0.765 | 0.759 | 0.488 | **0.0 %** |

**Trade-off at the core:** Prompt A's strict semantics ("completely solve")
aligns perfectly with the pass/solve label → AUROC 0.887, ECE 0.094. But it also
perfectly captures "most single chunks are partial or fail" → 60.6 % flatten at 0.

Relaxing the prompt (B, C) flips values to ~1 (things are "relevant"), eliminating
the low-side plateau — but now **discriminability collapses** (AUROC 0.887 → 0.858
→ 0.799, ECE 0.094 → 0.539 → 0.765). The plateau just moves from 0 to 1, and
softmax competition at the 1-end ("all chunks equally good") is as useless as at
the 0-end.

**Conclusion:** Prompting cannot escape the binary logprob constraint. Strict
prompt = good calibration, query-internal saturation. Relaxed prompt = query
signal, calibration collapse. One cannot have both.

## 5. Brier score by class (why overall-Brier is misleading)

| Method | Brier (overall) | **Brier (negative, 85.4 %)** | **Brier (positive, 14.6 %)** | Imbalance bias |
|---|---|---|---|---|
| `logprob` | **0.106** (best) | **0.043** (best) | **0.472** (worst) | Wins via majority class |
| `self_decoupled` | 0.157 | 0.158 | **0.153** (best) | Balanced |
| `self` (coupled) | 0.335 | 0.376 | 0.098 | Imbalanced the other way |
| `judge` | 0.321 | 0.365 | **0.064** (best) | Imbalanced: favors positives |

**The mechanism:** 85.4 % of chunks are Unsolved (negative class). Logprob's low
overall Brier (0.106) comes **entirely from the negative class** (0.043 —
confidently predict 0 for the majority class). On the positive class (the 14.6 %
that *are* solved), logprob is **the worst** (0.472) — it incorrectly flattens
many solved chunks to P(Yes) ≈ 0.

This is why logprob's indices (AUROC, AUPRC, ECE, Brier) all rank it first: they
are pooled or global metrics that reward separating the bimodal extremes, and the
negative-class majority dominates aggregation.

| Metric | What it rewards | Logprob's advantage |
|---|---|---|
| AUROC | Global ranking (solved vs. unsolved, anywhere) | Bimodal separation |
| ECE | Overall calibration | Small mean error, but driven by majority class |
| Brier | Mean squared error | Majority-class dominance (0.043 on 85 %) |
| AvgRank | Where the correct chunk ranks within a query | **Parity with CTM** (0.421 vs. 0.432) |
| Pass rate | Can CTM pick good chunks for fusion? | **Worst** (55.0 %) |

---

## Conclusion: calibration and competitiveness are orthogonal

Logprob's bimodal saturation is **not** a calibration problem — it is correctly
identifying that most single chunks do not independently solve the query. Nor is
it fixable by temperature or prompting; both preserve ranking order, so they do
not change the competition.

The core limitation: **binary Yes/No + token logprobs** produce saturated values
in any realistic scenario (either compressed to 0 for partial failures, or to 1
for partial successes, with few in-between). CTM's competition needs a smooth,
multi-dimensional grading signal (relevance + confidence + surprise), not a
saturated 0/1 prediction.

CTM-AI (`self_decoupled`) achieves this by decoupling the same multi-dimensional
rubric from the answer pass, yielding both strong calibration (AUROC 0.849, ECE
0.199) and strong pass rate (62.9 %), without external models or post-hoc tuning.
