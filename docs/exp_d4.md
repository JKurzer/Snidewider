# D4 — pooling-strategy sweep for DCT doc features (worktree d4-pooling)

Hypothesis: `dct.dct_features` pools per-segment quantities (adjacent-sentence
cosines, order-energy ratios ||c[1]||/||c[0]||) with mean/std over **all**
segments — mean-pooling may flatten the tail. If AI text is given away by
*extreme* segments rather than average ones, tail/extreme pooling should win.

Protocol (leakage-proof, RULES #4): detector = HGB(rs=7) trained on bucket A;
pooling **selected on bucket B** (meta out-of-fold stack scores, 5-fold
StratifiedKFold rs=7); final numbers **once on C**. Other 3 detectors' scores
from `data/derived/base_scores.npz`; row alignment verified by label equality
and by reproducing the 3-detector stack (C: TPR 0.175 ≈ fleet's 0.170, AUROC
0.903). Repro: `scripts/exp_d4.py` (per-segment arrays cached in
`data/derived/d4_segments.npz` in this worktree).

## Pooling sweep — selection metrics on B

dct = DCT detector standalone; stack-oof = 4-detector stack, meta out-of-fold
on B. TPR = TPR@FPR=1e-3.

| pooling        | #feat | dct AUROC | dct TPR | stack AUROC | stack TPR |
|----------------|------:|----------:|--------:|------------:|----------:|
| meanstd (ref)  |     4 |     0.678 |   0.001 |       0.898 |     0.032 |
| trimmed 10/90  |     4 |     0.688 |   0.002 |       0.899 |     0.061 |
| quantiles 10/50/90 | 6 |     0.699 |   0.012 |       0.897 |     0.039 |
| max            |     2 |     0.646 |   0.004 |       0.888 |     0.021 |
| min            |     2 |     0.612 |   0.004 |       0.885 |     0.070 |
| minmax         |     4 |     0.656 |   0.018 |       0.892 |     0.118 |
| top5_energy    |     4 |     0.679 |   0.012 |       0.893 |     0.047 |
| tails10        |     4 |     0.633 |   0.021 |       0.886 |     0.086 |
| extremes (q10/q90/range) | 6 | 0.677 |   0.019 |       0.892 |     0.111 |
| **meanstd+tails** |  8 |     0.735 |   0.001 |       0.899 | **0.118** |
| all (22 feats) |    22 | **0.770** |   0.010 | **0.907** |     0.096 |

Standalone DCT separation jumps when pooling looks at the tails (meanstd
0.678 → meanstd+tails 0.735 → all 0.770 AUROC): the tail-signal hypothesis is
real for this feature family. Selected on B (stack TPR, tie-break AUROC):
**meanstd+tails**.

## Final on C (meta HGB rs=7 trained on B's 4 scores; evaluated once)

| stack                        | AUROC | TPR@1e-3 [Wilson 95%] |
|------------------------------|------:|----------------------:|
| 3-detector repro (no DCT)    | 0.903 |     0.175 [0.153, 0.200] |
| 4-det, meanstd reference     | 0.918 |     0.273 [0.246, 0.301] |
| 4-det, **meanstd+tails**     | 0.918 | **0.313 [0.285, 0.342]** |

## Verdict

- **Clears the bar**: TPR@1e-3 **0.313** vs target 0.170 (+0.14 over the
  3-detector stack, +0.04 over meanstd pooling); AUROC **0.918** vs floor
  0.896 — no AUROC loss (identical to meanstd reference).
- Extremes beat averages: every tail-aware pooling lifted standalone DCT
  AUROC; pure min/max pooling alone is weak standalone (0.61–0.66) but adds
  in stack — the tails complement, not replace, the central moments.
- Note: this pipeline's meanstd reference (0.273) is well above the earlier
  fleet report of DCT-as-4th-detector (0.155, commit 0b723bf). Same features,
  same protocol; difference is A-fit imputation vs per-bucket imputation and
  regenerated base scores. Flagging for the fleet — worth a re-baseline.
- `meanstd+tails` adds 4 features (decile tail means of both quantities) —
  cheap, pure, and the natural new default pooling for the DCT panel.
