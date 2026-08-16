# FLEET F3 — incremental value of classical stats over the panel

HGB(rs=7) train A; B reference; C read once per arm. TPR@FPR=1e-2.

| arm | n_feat | AUROC B | TPR@1e-2 B | AUROC C | TPR@1e-2 C |
|---|---|---|---|---|---|
| panel | 89 | 0.896 | 0.196 [0.173, 0.222] | 0.892 | 0.165 [0.143, 0.189] |
| panel+stats | 111 | 0.926 | 0.369 [0.340, 0.399] | 0.922 | 0.328 [0.300, 0.358] |
| panel+full | 120 | 0.907 | 0.521 [0.490, 0.552] | 0.909 | 0.394 [0.364, 0.425] |
| stats+full | 31 | 0.895 | 0.498 [0.467, 0.529] | 0.895 | 0.370 [0.341, 0.400] |

## Spearman |rho| on C: coverage stars vs panel neighbours

- cov2_contrast: shape_dct_run_rand_stdev 0.48, qg_mid_qgram_mean 0.37, shape_dct_run_rand_mean 0.31, dct_bands_c0_std 0.30
- cov3_contrast: qg_mid_qgram_mean 0.40, dct_bands_c0_std 0.33, qg_mid_ck2_mean 0.32, shape_dct_run_rand_mean 0.32
- cov5_contrast: qg_mid_qgram_mean 0.42, qg_mid_ck2_mean 0.37, shape_dct_run_rand_mean 0.33, ex_contrast_min 0.26
