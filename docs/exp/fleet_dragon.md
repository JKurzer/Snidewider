# FLEET DRAGON — Kolmogorov proxies (Navarro ladder)

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| kol_lcp_mean | 0.518 | 0.506 | 0.161 | 0.135 |
| kol_lcp_p90 | 0.519 | 0.510 | 0.056 | 0.058 |
| kol_lcp_max | 0.506 | 0.519 | 0.090 | 0.088 |
| kol_repair_rules_rate | 0.521 | 0.529 | 0.052 | 0.054 |
| kol_repair_total_rate | 0.533 | 0.522 | 0.145 | 0.161 |
| kol_sam_trans_rate | 0.531 | 0.528 | 0.094 | 0.106 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C |
|---|---|---|---|
| panel153 | 153 | 0.994 | 0.921 [0.903,0.936] |
| panel+dragon159 | 159 | 0.993 | 0.912 [0.893,0.928] |
| dragon6 alone | 6 | 0.773 | 0.177 [0.155,0.202] |
