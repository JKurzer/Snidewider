# FLEET L — CSA compressibility-profile shape features

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| shape_stdev | 0.523 | 0.509 | 0.005 | 0.009 |
| shape_iqr | 0.539 | 0.508 | 0.045 | 0.033 |
| shape_range | 0.518 | 0.514 | 0.005 | 0.010 |
| shape_arc | 0.532 | 0.560 | 0.008 | 0.011 |
| shape_maxmin | 0.500 | 0.517 | 0.023 | 0.034 |
| shape_cv | 0.508 | 0.526 | 0.025 | 0.039 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C |
|---|---|---|---|
| panel156 | 156 | 0.994 | 0.918 [0.899,0.933] |
| panel+shape162 | 162 | 0.993 | 0.921 [0.903,0.936] |
| shape6 alone | 6 | 0.700 | 0.066 [0.052,0.083] |
