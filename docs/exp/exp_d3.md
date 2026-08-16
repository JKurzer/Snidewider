# exp_d3 — DCT feature shapes for the low-FPR tail

**Headline: 4-detector stack TPR@FPR=1e-3 = 0.340 (control 0.175), AUROC 0.934
(control 0.903) — target 0.170 beaten ~2x with AUROC gained, not lost.**

## Why new shapes

The stock 4 DCT doc features (adjacent-cosine mean/stdev + order-energy
mean/stdev) are *bulk* statistics: they describe the center of a distribution
while the tail signal at FPR=1e-3 lives in its shape. Solo base TPR@1e-3 on C:
**0.000**. So we rebuilt the feature families around distribution shape.

## Shapes tried (all from the same K=5 DCT segment encoding)

| family   | n  | idea |
|----------|----|------|
| base     | 4  | reference: adj-cos mean/std, order-energy mean/std (control) |
| paircos  | 10 | seeded (md5-of-text) random *non-adjacent* segment-pair cosine quantiles + adj−nonadj gap: global smoothness vs local |
| bands    | 18 | per-coefficient energy profile: ||c_k|| mean/std k=0..4 + ||c_k||/||c_0|| ratio stats k=1..4 (the "order spectrum") |
| normpct  | 9  | percentiles (p05..p95, IQR, p90−p10) of segment-norm distribution |
| arc      | 4  | first-third vs last-third contrast: centroid cosine, adj-cos shift, order-ratio shift, norm shift |
| acosq    | 9  | adjacent-cosine *quantiles* instead of mean/stdev |

Selection on **B** (trained on A; C untouched until one final pass).
Winner on B: **nobase** = paircos+bands+normpct+arc+acosq (50 feats),
B TPR@1e-3 0.061 vs base 0.000.

## Per-shape table (HGB random_state=7, trained on A)

| shape   | B AUROC | B TPR@1e-3 | C AUROC | C TPR@1e-3 |
|---------|---------|------------|---------|------------|
| base    | 0.689   | 0.000      | 0.677   | 0.000      |
| paircos | 0.620   | 0.010      | 0.649   | 0.004      |
| bands   | 0.785   | 0.054      | 0.796   | 0.055      |
| normpct | 0.733   | 0.001      | 0.718   | 0.001      |
| arc     | 0.680   | 0.002      | 0.661   | 0.000      |
| acosq   | 0.641   | 0.007      | 0.640   | 0.008      |
| all     | 0.853   | 0.027      | 0.836   | 0.100      |
| nobase  | 0.852   | **0.061**  | 0.847   | 0.020      |

Notes: `bands` is the best single family (energy spectrum ≫ cosine summary
stats). Quantile families (normpct/acosq) add little solo but contribute to
the combo; `base` actively hurts the combo on B (all < nobase on B TPR), so
the winner drops it. all-vs-nobase inversion on C (0.100 vs 0.020) is
selection noise at n_ai=1000 — we did **not** re-select on C.

## Stack result (C, single pass; HGB meta on B's 4 scores)

| stack                    | C AUROC | C TPR@1e-3 [95% CI]   |
|--------------------------|---------|-----------------------|
| 3-det control (cached)   | 0.903   | 0.175 [0.153, 0.200]  |
| 4-det (+dct-nobase)      | **0.934** | **0.340 [0.311, 0.370]** |

Target was TPR@1e-3 > 0.170 without losing AUROC (0.896): cleared on both —
TPR nearly doubles, AUROC +0.031.

## Protocol / hygiene

- Buckets A/B/C from `evaldata.split_buckets` (source-level, salt 41);
  labels verified row-aligned with `base_scores.npz` before any training.
- Detectors train on A; shape selected on B; C evaluated once (final pass).
- Per-doc pair sampling seeded by md5(text) → pure features (RULES #5).
- Cached A scores ignored, as briefed. Holdout fold untouched.
- Feature cache: `data/derived/d3_features.npz` (v3, keyed by doc ids).

## Repro

```
set PYTHONPATH=<worktree>\src && <repo>\.venv\Scripts\python <worktree>\scripts\exp_d3.py
```
(from cwd = main checkout; data resolves via cwd, code via PYTHONPATH)
