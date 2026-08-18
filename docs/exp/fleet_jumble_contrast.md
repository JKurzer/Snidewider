# FLEET V — jumble contrast (positional loading) solo

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| oct_order_load | 0.558 | 0.560 | 0.057 | 0.053 |
| reuse_order_load | 0.536 | 0.506 | 0.021 | 0.000 |
| zipf_struct_load | 0.522 | 0.527 | 0.103 | 0.114 |
| cond_order_load | 0.510 | 0.511 | 0.060 | 0.082 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |
|---|---|---|---|---|
| panel250 | 250 | 0.9932 | 0.933 [0.916,0.947] | 0.778 |
| panel+254 | 254 | 0.9930 | 0.909 [0.890,0.925] | 0.779 |
| jc4 alone | 4 | 0.6916 | 0.135 [0.115,0.158] | 0.113 |
