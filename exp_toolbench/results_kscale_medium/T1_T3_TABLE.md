
### T1 — K ∈ {1,2,4,8,16} (Pass Rate per set, Avg API Calls, Latency, Active Links)

| K | I2-Cat Pass↑ | I2-Inst Pass↑ | I3-Inst Pass↑ | Avg API Calls↓ | Avg Latency↓ | Avg Active Links |
|---|---|---|---|---|---|---|
| 1 | 47.4 | 25.8 | 65.6 | 2.30 | 47.33 | 0.00 |
| 2 | 52.8 | 50.8 | 57.5 | 4.26 | 86.40 | 0.16 |
| 4 | 74.2 | 44.2 | 79.2 | 7.35 | 108.32 | 0.60 |
| 8 | 60.0 | 49.2 | 82.5 | 14.02 | 147.30 | 1.27 |

### T3 — K ∈ {2,4,8,16,32} (Pass Rate per set, Latency, Model Calls, Active Links)

| K | I2-Cat↑ | I2-Inst↑ | I3-Inst↑ | Latency↓ | # Model Calls↓ | # Active Links |
|---|---|---|---|---|---|---|
| 1 | 47.4 | 25.8 | 65.6 | 47.33 | 5.61 | 0.00 |
| 2 | 52.8 | 50.8 | 57.5 | 86.40 | 9.84 | 0.16 |
| 4 | 74.2 | 44.2 | 79.2 | 108.32 | 18.13 | 0.60 |
| 8 | 60.0 | 49.2 | 82.5 | 147.30 | 33.60 | 1.27 |

### Coverage (queries with metrics per K × test set)

| K | G2_category | G2_instruction | G3_instruction | avg #proc |
|---|---|---|---|---|
| 1 | 19 | 20 | 16 | 1.00 |
| 2 | 18 | 20 | 20 | 1.76 |
| 4 | 20 | 20 | 20 | 3.60 |
| 8 | 20 | 20 | 20 | 7.17 |
