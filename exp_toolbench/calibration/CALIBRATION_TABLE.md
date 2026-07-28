
chunks with label: 926 | positive (Solved): 135 (14.6%)

| Confidence estimator                         | AUROC↑ | AUPRC↑ |  ECE↓ | Brier↓ | AvgRank↓ |
|----------------------------------------------|--------|--------|-------|--------|----------|
| Self-reported score (CTM weight)             |  0.731 |  0.280 | 0.452 |  0.335 |    0.485 |
| External LLM-as-a-judge (decoupled Qwen3-8B) |  0.824 |  0.410 | 0.441 |  0.321 |    0.488 |
| Self-prompt Yes/No logprobs                  |  0.886 |  0.612 | 0.097 |  0.106 |    0.422 |

