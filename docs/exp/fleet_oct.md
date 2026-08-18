# FLEET R — reduced pack (cv + initial + bwt + oct)

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| cv_cc_rate | 0.536 | 0.533 | 0.029 | 0.037 |
| cv_cv_rate | 0.529 | 0.522 | 0.014 | 0.028 |
| cv_vc_rate | 0.533 | 0.528 | 0.015 | 0.026 |
| cv_vv_rate | 0.513 | 0.501 | 0.034 | 0.043 |
| initial_char_entropy | 0.579 | 0.596 | 0.093 | 0.114 |
| bwt_run_max | 0.569 | 0.564 | 0.088 | 0.080 |
| bwt_run_mean | 0.534 | 0.525 | 0.196 | 0.178 |
| bwt_run_p90 | 0.503 | 0.508 | 0.001 | 0.083 |
| bwt_run_entropy | 0.523 | 0.520 | 0.145 | 0.156 |
| bwt_char_entropy | 0.598 | 0.604 | 0.042 | 0.066 |
| oct_repeat_rate | 0.595 | 0.589 | 0.000 | 0.151 |
| oct_repeat_abs | 0.629 | 0.618 | 0.000 | 0.131 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C |
|---|---|---|---|
| panel226 | 268 | 0.994 | 0.914 [0.895,0.930] |
| panel+238 | 280 | 0.992 | 0.899 [0.879,0.916] |
| pack12 alone | 12 | 0.808 | 0.196 [0.173,0.222] |
