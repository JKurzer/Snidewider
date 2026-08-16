# FLEET B — <test> vs <general> mega-reference sweep

references: pooled bucket-A profiles per class (one native diff per side). selection on B, confirmation on C; calib fit on B humans. TPR@FPR=1e-2. DEV ONLY.

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C | max |rho| vs panel |
|---|---|---|---|---|---|
| q5_d_ai | 0.658 | 0.668 | 0.000 | 0.000 | 0.69 |
| q3_d_ai | 0.622 | 0.632 | 0.000 | 0.000 | 0.67 |
| q5_d_hu | 0.623 | 0.631 | 0.000 | 0.000 | 0.70 |
| q5_calib_pct | 0.622 | 0.631 | 0.000 | 0.000 | 0.70 |
| q3_d_hu | 0.609 | 0.618 | 0.000 | 0.000 | 0.67 |
| q3_calib_pct | 0.608 | 0.618 | 0.000 | 0.000 | 0.67 |
| q3_ratio | 0.595 | 0.602 | 0.000 | 0.000 | 0.68 |
| q3_contrast | 0.595 | 0.602 | 0.000 | 0.000 | 0.68 |
| q5_ratio | 0.581 | 0.586 | 0.000 | 0.000 | 0.72 |
| q5_contrast | 0.581 | 0.586 | 0.000 | 0.000 | 0.72 |

## panel reference on C (existing features)

- ex_hu_mean: AUROC 0.655, TPR@1e-2 0.049
- ex_hu_min: AUROC 0.574, TPR@1e-2 0.000
- ex_contrast_p10: AUROC 0.787, TPR@1e-2 0.249
- qg_mid_qgram_mean: AUROC 1.000, TPR@1e-2 0.000
