# FLEET I — per-feature bug audit (156 cols, dev cache)

## flagged features

| feature | flags |
|---|---|
| rel_short_range_mean | ORIENT-FLIP signs=[np.float64(1.0), np.float64(-1.0), np.float64(-1.0)] |
| rel_short_range_stdev | ORIENT-FLIP signs=[np.float64(1.0), np.float64(-1.0), np.float64(-1.0)] |
| qg_mid_ck2_mean | nanA=0.83; nanB=0.81; nanC=0.81 |
| qg_mid_ck2_stdev | nanA=0.83; nanB=0.81; nanC=0.81 |
| qg_mid_qgram_mean | nanA=0.83; nanB=0.81; nanC=0.81 |
| qg_mid_qgram_stdev | nanA=0.83; nanB=0.81; nanC=0.81 |
| ex_ai_mean | ORIENT-FLIP signs=[np.float64(1.0), np.float64(-1.0), np.float64(1.0)] |
| ex_hu_mean_raw | LENGTH-GHOST A rho=0.951; LENGTH-GHOST B rho=0.953; LENGTH-GHOST C rho=0.951 |
| dct_arc_adj | ORIENT-FLIP signs=[np.float64(-1.0), np.float64(1.0), np.float64(-1.0)] |
| dct_arc_norm | ORIENT-FLIP signs=[np.float64(-1.0), np.float64(1.0), np.float64(-1.0)] |
| dct_arc_ratio | ORIENT-FLIP signs=[np.float64(1.0), np.float64(-1.0), np.float64(1.0)] |
| dct_bands_c3_mean | ORIENT-FLIP signs=[np.float64(1.0), np.float64(1.0), np.float64(-1.0)] |
| dct_bands_c4_mean | ORIENT-FLIP signs=[np.float64(1.0), np.float64(-1.0), np.float64(-1.0)] |
| dct_paircos_gap | ORIENT-FLIP signs=[np.float64(-1.0), np.float64(-1.0), np.float64(1.0)] |
| shape_skeleton_step_mean | ORIENT-FLIP signs=[np.float64(1.0), np.float64(1.0), np.float64(-1.0)] |
| shape_skeleton_step_stdev | ORIENT-FLIP signs=[np.float64(-1.0), np.float64(1.0), np.float64(-1.0)] |
| shape_dct_run_step_mean | nanA=0.68; nanB=0.68; nanC=0.68 |
| shape_dct_run_step_stdev | nanA=0.68; nanB=0.68; nanC=0.68 |
| shape_dct_run_rand_mean | nanA=0.94; nanB=0.93; nanC=0.93 |
| shape_dct_run_rand_stdev | nanA=0.94; nanB=0.93; nanC=0.93 |
| stat_simpson_d | ORIENT-FLIP signs=[np.float64(1.0), np.float64(-1.0), np.float64(1.0)] |
| cov5_hu | ORIENT-FLIP signs=[np.float64(1.0), np.float64(1.0), np.float64(-1.0)] |
| col_slope_ratio | ORIENT-FLIP signs=[np.float64(-1.0), np.float64(1.0), np.float64(1.0)] |
| col_bigram_repeat_mass | ORIENT-FLIP signs=[np.float64(1.0), np.float64(-1.0), np.float64(-1.0)] |

## near-duplicate pairs (|rho| > 0.9999 on C)

| feature a | feature b | rho |
|---|---|---|
| rel_qgram_distinct_ratio | qg_q3_distinct_ratio | +1.00000 |
| rel_qgram_entropy | qg_q3_entropy | +1.00000 |
| rel_qgram_max_share | qg_q3_max_share | +1.00000 |
| rel_qgram_repeat_frac | qg_q3_repeat_frac | +1.00000 |
| stat_hapax | col_spec_k1 | +1.00000 |
| stat_dis | col_spec_k2 | +1.00000 |
| col_cond_entropy | chr_bigram_cond_entropy | +1.00000 |

## sentinels (pure packs on edge texts)

- empty: ok (no crash)
- one-word: ok (no crash)
- same-char: ok (no crash)
- punct-only: ok (no crash)
- short-eng: ok (no crash)
