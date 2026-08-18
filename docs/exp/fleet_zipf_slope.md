# FLEET T — jumble-gradient derivatives (solo)

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| js_zipf_slope_rate | 0.537 | 0.514 | 0.039 | 0.054 |
| js_zipf_slope_drop | 0.528 | 0.506 | 0.039 | 0.053 |
| js_zipf_r2_rate | 0.569 | 0.551 | 0.138 | 0.128 |
| js_zipf_r2_drop | 0.552 | 0.537 | 0.120 | 0.125 |
| js_bg_ent_rate | 0.513 | 0.500 | 0.094 | 0.080 |
| js_bg_ent_drop | 0.515 | 0.504 | 0.081 | 0.095 |
| js_cond_ent_rate | 0.513 | 0.500 | 0.094 | 0.080 |
| js_cond_ent_drop | 0.515 | 0.504 | 0.081 | 0.095 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |
|---|---|---|---|---|
| panel250 | 250 | 0.9932 | 0.933 [0.916,0.947] | 0.778 |
| panel+258 | 258 | 0.9930 | 0.913 [0.894,0.929] | 0.778 |
| js8 alone | 8 | 0.7400 | 0.195 [0.172,0.221] | 0.141 |
