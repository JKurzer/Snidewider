# Per-feature performance on bucket C (source-disjoint, evaluated once)

AUROC is direction-corrected (direction column gives the raw sign). TPR at target FPR=1e-3 with achieved FPR + Wilson CI. coverage = fraction of docs with a finite value.

| feature | coverage | auroc | direction | sep | tpr | ci | fpr_achieved |
|---|---|---|---|---|---|---|---|
| qg_mid_ck2_mean | 0.092 | 0.978 | ai>lower | 0.978 | 0.419 | [0.284, 0.567] | 0.000 |
| qg_mid_qgram_mean | 0.092 | 0.974 | ai>lower | 0.974 | 0.395 | [0.264, 0.544] | 0.000 |
| ex_contrast_p10 | 1.000 | 0.787 | ai>lower | 0.787 | 0.156 | [0.135, 0.180] | 0.000 |
| shape_dct_run_step_stdev | 0.428 | 0.764 | ai>lower | 0.764 | 0.000 | [0.000, 0.007] | 0.000 |
| ex_contrast_mean | 1.000 | 0.723 | ai>lower | 0.723 | 0.157 | [0.136, 0.181] | 0.000 |
| ex_ai_mean | 1.000 | 0.694 | ai>lower | 0.694 | 0.122 | [0.103, 0.144] | 0.000 |
| ex_contrast_min | 1.000 | 0.692 | ai>lower | 0.692 | 0.147 | [0.126, 0.170] | 0.000 |
| dct_normpct_spread | 0.999 | 0.692 | ai>lower | 0.692 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_bands_c0_std | 0.960 | 0.685 | ai>lower | 0.685 | 0.015 | [0.009, 0.025] | 0.000 |
| dct_normpct_iqr | 0.999 | 0.683 | ai>lower | 0.683 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_bands_r2_std | 0.960 | 0.683 | ai>lower | 0.683 | 0.000 | [-0.000, 0.004] | 0.000 |
| ex_ai_p10 | 1.000 | 0.682 | ai>lower | 0.682 | 0.112 | [0.094, 0.133] | 0.000 |
| ex_ai_min | 1.000 | 0.669 | ai>lower | 0.669 | 0.103 | [0.086, 0.123] | 0.000 |
| dct_bands_r4_std | 0.960 | 0.669 | ai>lower | 0.669 | 0.003 | [0.001, 0.009] | 0.000 |
| dct_bands_r3_std | 0.960 | 0.667 | ai>lower | 0.667 | 0.002 | [0.001, 0.008] | 0.000 |
| ex_hu_mean | 1.000 | 0.655 | ai>lower | 0.655 | 0.007 | [0.003, 0.014] | 0.000 |
| dct_bands_r1_std | 0.960 | 0.652 | ai>lower | 0.652 | 0.002 | [0.001, 0.008] | 0.000 |
| qg_mid_ck2_stdev | 0.092 | 0.644 | ai>higher | 0.644 | 0.000 | [0.000, 0.082] | 0.000 |
| qg_mid_qgram_stdev | 0.092 | 0.644 | ai>higher | 0.644 | 0.000 | [0.000, 0.082] | 0.000 |
| rel_midrange_mean | 1.000 | 0.639 | ai>lower | 0.639 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_normpct_p10 | 0.999 | 0.638 | ai>higher | 0.638 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_normpct_p25 | 0.999 | 0.636 | ai>higher | 0.636 | 0.000 | [0.000, 0.004] | 0.000 |
| ex_hu_p10 | 1.000 | 0.631 | ai>lower | 0.631 | 0.026 | [0.018, 0.038] | 0.000 |
| dct_acosq_iqr | 0.960 | 0.631 | ai>lower | 0.631 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_normpct_p05 | 0.999 | 0.627 | ai>higher | 0.627 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_acosq_p25 | 0.960 | 0.626 | ai>higher | 0.626 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_paircos_iqr | 0.954 | 0.623 | ai>lower | 0.623 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_acosq_spread | 0.960 | 0.621 | ai>lower | 0.621 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_paircos_spread | 0.954 | 0.620 | ai>lower | 0.620 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_bands_c4_std | 0.960 | 0.616 | ai>lower | 0.616 | 0.001 | [0.000, 0.006] | 0.000 |
| dct_acosq_p10 | 0.960 | 0.615 | ai>higher | 0.615 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_paircos_p05 | 0.954 | 0.612 | ai>higher | 0.612 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_paircos_p10 | 0.954 | 0.611 | ai>higher | 0.611 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_acosq_p05 | 0.960 | 0.610 | ai>higher | 0.610 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_paircos_p25 | 0.954 | 0.609 | ai>higher | 0.609 | 0.000 | [0.000, 0.004] | 0.000 |
| rel_qgram_distinct_ratio | 1.000 | 0.607 | ai>lower | 0.607 | 0.000 | [0.000, 0.004] | 0.000 |
| qg_q3_distinct_ratio | 1.000 | 0.607 | ai>lower | 0.607 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_acosq_p50 | 0.960 | 0.606 | ai>higher | 0.606 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_bands_r2_mean | 0.960 | 0.604 | ai>lower | 0.604 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_drift_cos | 0.954 | 0.602 | ai>higher | 0.602 | 0.000 | [0.000, 0.004] | 0.000 |
| qg_q3_repeat_frac | 1.000 | 0.595 | ai>higher | 0.595 | 0.003 | [0.001, 0.009] | 0.000 |
| rel_qgram_repeat_frac | 1.000 | 0.595 | ai>higher | 0.595 | 0.003 | [0.001, 0.009] | 0.000 |
| dct_bands_r4_mean | 0.960 | 0.593 | ai>lower | 0.593 | 0.000 | [-0.000, 0.004] | 0.000 |
| ex_hu_mean_raw | 1.000 | 0.593 | ai>higher | 0.593 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_normpct_p50 | 0.999 | 0.592 | ai>higher | 0.592 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_paircos_p50 | 0.954 | 0.592 | ai>higher | 0.592 | 0.001 | [0.000, 0.006] | 0.000 |
| dct_bands_r3_mean | 0.960 | 0.591 | ai>lower | 0.591 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_bands_r1_mean | 0.960 | 0.586 | ai>lower | 0.586 | 0.000 | [-0.000, 0.004] | 0.000 |
| ex_hu_min | 1.000 | 0.574 | ai>lower | 0.574 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_bands_c0_mean | 0.960 | 0.571 | ai>higher | 0.571 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_bands_c3_std | 0.960 | 0.570 | ai>lower | 0.570 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_bands_c2_std | 0.960 | 0.569 | ai>lower | 0.569 | 0.002 | [0.001, 0.008] | 0.000 |
| qg_q2_entropy | 1.000 | 0.566 | ai>lower | 0.566 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_acosq_p75 | 0.960 | 0.564 | ai>higher | 0.564 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_paircos_p75 | 0.954 | 0.564 | ai>higher | 0.564 | 0.000 | [0.000, 0.004] | 0.000 |
| shape_dct_run_step_mean | 0.428 | 0.552 | ai>lower | 0.552 | 0.006 | [0.002, 0.017] | 0.000 |
| shape_skeleton_rand_mean | 1.000 | 0.546 | ai>lower | 0.546 | 0.001 | [0.000, 0.006] | 0.000 |
| qg_q5_top10_share | 1.000 | 0.545 | ai>lower | 0.545 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_normpct_p95 | 0.999 | 0.544 | ai>lower | 0.544 | 0.000 | [0.000, 0.004] | 0.000 |
| rel_midrange_stdev | 1.000 | 0.544 | ai>lower | 0.544 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_paircos_p90 | 0.954 | 0.541 | ai>higher | 0.541 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_acosq_p90 | 0.960 | 0.539 | ai>higher | 0.539 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_drift_norm | 0.954 | 0.539 | ai>higher | 0.539 | 0.000 | [0.000, 0.004] | 0.000 |
| shape_skeleton_step_mean | 1.000 | 0.533 | ai>higher | 0.533 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_normpct_p90 | 0.999 | 0.533 | ai>lower | 0.533 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_acosq_p95 | 0.960 | 0.533 | ai>higher | 0.533 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_bands_c4_mean | 0.960 | 0.532 | ai>lower | 0.532 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_paircos_p95 | 0.954 | 0.531 | ai>higher | 0.531 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_bands_c2_mean | 0.960 | 0.530 | ai>lower | 0.530 | 0.000 | [-0.000, 0.004] | 0.000 |
| qg_q3_entropy | 1.000 | 0.530 | ai>higher | 0.530 | 0.000 | [0.000, 0.004] | 0.000 |
| rel_qgram_entropy | 1.000 | 0.530 | ai>higher | 0.530 | 0.000 | [0.000, 0.004] | 0.000 |
| qg_short_ck2_mean | 1.000 | 0.526 | ai>lower | 0.526 | 0.001 | [0.000, 0.006] | 0.000 |
| shape_skeleton_rand_stdev | 1.000 | 0.526 | ai>lower | 0.526 | 0.002 | [0.001, 0.007] | 0.000 |
| dct_bands_c1_std | 0.960 | 0.526 | ai>lower | 0.526 | 0.001 | [0.000, 0.006] | 0.000 |
| dct_drift_ratio | 0.954 | 0.526 | ai>lower | 0.526 | 0.000 | [0.000, 0.004] | 0.000 |
| rel_short_range_mean | 1.000 | 0.526 | ai>lower | 0.526 | 0.001 | [0.000, 0.006] | 0.000 |
| shape_skeleton_step_stdev | 1.000 | 0.525 | ai>lower | 0.525 | 0.003 | [0.001, 0.009] | 0.000 |
| qg_short_ck2_stdev | 1.000 | 0.523 | ai>lower | 0.523 | 0.003 | [0.001, 0.009] | 0.000 |
| rel_short_range_stdev | 1.000 | 0.523 | ai>lower | 0.523 | 0.003 | [0.001, 0.009] | 0.000 |
| dct_bands_c1_mean | 0.960 | 0.520 | ai>higher | 0.520 | 0.000 | [-0.000, 0.004] | 0.000 |
| dct_normpct_p75 | 0.999 | 0.520 | ai>higher | 0.520 | 0.000 | [0.000, 0.004] | 0.000 |
| ex_ai_mean_raw | 1.000 | 0.518 | ai>higher | 0.518 | 0.000 | [0.000, 0.004] | 0.000 |
| rel_qgram_max_share | 1.000 | 0.518 | ai>lower | 0.518 | 0.000 | [0.000, 0.004] | 0.000 |
| qg_q3_max_share | 1.000 | 0.518 | ai>lower | 0.518 | 0.000 | [0.000, 0.004] | 0.000 |
| dct_drift_adj | 0.954 | 0.513 | ai>higher | 0.513 | 0.001 | [0.000, 0.006] | 0.000 |
| dct_bands_c3_mean | 0.960 | 0.504 | ai>lower | 0.504 | 0.001 | [0.000, 0.006] | 0.000 |
| dct_paircos_gap | 0.954 | 0.500 | ai>higher | 0.500 | 0.000 | [0.000, 0.004] | 0.000 |
| shape_dct_run_rand_mean | 0.045 | nan | nan | nan | nan | nan | nan |
| shape_dct_run_rand_stdev | 0.045 | nan | nan | nan | nan | nan | nan |

