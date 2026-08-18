# FLEET X — covering number + within-doc density (solo)

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C | TPR@1e-3 B | TPR@1e-3 C |
|---|---|---|---|---|---|---|
| cover_balls | 0.570 | 0.584 | 0.086 | 0.163 | 0.028 | 0.066 |
| cover_balls_rate | 0.533 | 0.547 | 0.074 | 0.115 | 0.025 | 0.050 |
| wd_density | 0.558 | 0.576 | 0.079 | 0.107 | 0.008 | 0.058 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |
|---|---|---|---|---|
| panel250 | 250 | 0.9932 | 0.933 [0.916,0.947] | 0.778 |
| panel+253 | 253 | 0.9924 | 0.907 [0.887,0.923] | 0.738 |
| cover3 alone | 3 | 0.6491 | 0.148 [0.127,0.171] | 0.117 |
