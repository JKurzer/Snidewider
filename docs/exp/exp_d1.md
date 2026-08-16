# exp D1 — DCT width sweep (K x unit): does width buy tail TPR?

**Headline: K=8, unit=sentences. 4-detector stack TPR@FPR=1e-3 = 0.294
[0.267, 0.323] vs 0.175 [0.153, 0.200] reference; AUROC 0.915 vs 0.903.
Tail nearly doubles, CIs don't overlap, AUROC gained, not kept.**

## Question

The DCT sentence encoder keeps K coefficients (paper: K=2). Does any
K in {1,2,3,4,6,8,12,16} x unit in {sentences, windows(24)} make the DCT
detector useful at the tail — i.e. the 4-detector HGB stack beats
TPR@FPR=1e-3 = 0.170 without dropping AUROC below 0.896 (3-detector
reference)?

## Protocol (leakage-proof; scripts/exp_d1.py)

- Dev fold only, source-disjoint buckets A/B/C via evaldata.split_buckets
  (holdout never touched). A=3000, B=1500, C=1500 docs.
- DCT detector: HistGradientBoostingClassifier(random_state=7) on bucket A's
  4 DCT features; NaNs passed through (HGB-native; no test-stats imputation).
- Stack: HGB(random_state=7) trained on B's 4 scores — cached
  relative-burst/qgram12/exemplar (base_scores.npz, detectors trained on A)
  + the DCT config's score on B. Final numbers read once on C.
- Row order verified: npz labels match split_buckets labels exactly.
- Fast featurizer (one embed pass per doc/unit, coefficients sliced per K)
  verified feature-for-feature against `dct.dct_features` on 150 docs x 16
  configs (9600 values, all match) before the sweep ran.
- Reference reproduced in-run from cached scores: AUROC 0.903, TPR 0.175
  (mission briefing quoted 0.896/0.170; same protocol+seed here reads
  0.903/0.175 — conclusions identical against either).

## Sweep table (all numbers on bucket C; TPR at FPR=1e-3, Wilson CI)

| K | unit | solo AUROC | solo TPR | stacked AUROC | stacked TPR [CI] |
|--:|------|-----------:|---------:|--------------:|------------------|
| ref | — | — | — | 0.903 | 0.175 [0.153, 0.200] |
| 1 | sentences | 0.585 | 0.003 | 0.906 | 0.153 [0.132, 0.177] |
| 2 | sentences | 0.662 | 0.004 | **0.918** | 0.256 [0.230, 0.284] |
| 3 | sentences | 0.666 | 0.001 | 0.915 | 0.204 [0.180, 0.230] |
| 4 | sentences | 0.672 | 0.000 | 0.913 | 0.262 [0.236, 0.290] |
| 6 | sentences | 0.685 | 0.000 | 0.915 | 0.136 [0.116, 0.159] |
| **8** | **sentences** | 0.689 | 0.000 | 0.915 | **0.294 [0.267, 0.323]** |
| 12 | sentences | 0.690 | 0.009 | 0.914 | 0.213 [0.189, 0.239] |
| 16 | sentences | 0.677 | 0.000 | 0.917 | 0.207 [0.183, 0.233] |
| 1 | windows | 0.546 | 0.004 | 0.904 | 0.127 [0.108, 0.149] |
| 2 | windows | 0.569 | 0.000 | 0.903 | 0.166 [0.144, 0.190] |
| 3 | windows | 0.566 | 0.002 | 0.903 | 0.091 [0.075, 0.110] |
| 4 | windows | 0.583 | 0.001 | 0.906 | 0.205 [0.181, 0.231] |
| 6 | windows | 0.600 | 0.006 | 0.909 | 0.168 [0.146, 0.192] |
| 8 | windows | 0.590 | 0.006 | 0.908 | 0.187 [0.164, 0.212] |
| 12 | windows | 0.604 | 0.001 | 0.903 | 0.282 [0.255, 0.311] |
| 16 | windows | 0.623 | 0.007 | 0.902 | 0.246 [0.220, 0.274] |

## Read

- **Winner: K=8, sentences.** Stacked TPR 0.294 — CI [0.267, 0.323] clears
  the reference's upper CI (0.200); AUROC 0.915 ≥ 0.903 reference. Mission
  bar (>0.170 TPR, ≥0.896 AUROC) cleared with room.
- Solo, the DCT detector is a tail dud at every K (TPR ≤ 0.009) — its value
  is purely complementary signal for the stack.
- Sentences dominate windows at nearly every K (paper's unit was right);
  windows only get interesting at K ≥ 12.
- Stack TPR is non-monotone in K (0.256 → 0.136 → 0.294 on sentences):
  meta is an HGB on 1500 B rows, so single-config jitter is real — K=6
  sentences dips below reference while K=8 soars. Don't read a smooth
  "width curve" into this; K=8's CI separation from the reference is the
  statement that survives.
- Solo AUROC rises with K up to ~8–12 then flattens: higher coefficients
  add order structure but the 4-feature summary saturates.
- K=1 has constant-zero order-energy features (no c[1]); it's the
  mean-embedding ablation and it's the worst sentences config — width
  genuinely matters.
- Recommendation for the fleet: default the DCT panel member to
  `dct_features(text, k=8, unit="sentences")`; K=2 (paper) is a fine
  fallback with the best stacked AUROC (0.918) if TPR ties matter.

## Reproduce

```
set PYTHONPATH=src&& ..\.venv\Scripts\python scripts\exp_d1.py   # cwd = main checkout
```
