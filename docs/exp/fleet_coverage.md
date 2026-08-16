# FLEET F2 — reference n-gram coverage (membership, not distance)

refs from A; select B, confirm C. TPR@FPR=1e-2. DEV ONLY.

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| cov2_contrast | 0.844 | 0.850 | 0.318 | 0.288 |
| cov3_contrast | 0.842 | 0.841 | 0.443 | 0.359 |
| cov5_contrast | 0.745 | 0.748 | 0.268 | 0.240 |
| cov2_ai | 0.709 | 0.712 | 0.192 | 0.138 |
| cov3_ai | 0.706 | 0.705 | 0.174 | 0.098 |
| cov5_ai | 0.686 | 0.685 | 0.153 | 0.089 |
| cov2_hu | 0.553 | 0.556 | 0.034 | 0.027 |
| cov3_hu | 0.535 | 0.529 | 0.028 | 0.023 |
| cov5_hu | 0.505 | 0.504 | 0.021 | 0.000 |
