# D5 — DCT score calibration for the 4-detector stack

**Headline: with the fleet's cached base scores, the 4-detector hgb stack does NOT
suffer the reported tail dilution — raw DCT gives TPR@1e-3 = 0.234 (3-det: 0.175).
Isotonic-calibrating the DCT score on B is the only variant that moves the corner
further: 0.249 [0.223, 0.277], AUROC 0.915. The 0.170→0.155 regression was a
code-drift artifact, not a DCT calibration problem.**

## Protocol (RULES #4/#3)

- Source-disjoint buckets A/B/C (3000/1500/1500 docs) via `evaldata.split_buckets`
  (salt 41, identical to the fleet). DCT detector = `HistGradientBoostingClassifier(7)`
  trained on A; calibrators fit on **B only**; all numbers on untouched C.
- Base scores (relative-burst, qgram12, exemplar) from `data/derived/base_scores.npz`
  (cached A scores ignored — in-sample). Bucket labels asserted equal to the cache.
- DCT = `dct.dct_features` defaults (k=2, sentences), NaNs imputed with A column means.
- Metas trained on B's (possibly calibrated) score matrix: `logreg` (StandardScaler +
  LR) and `hgb` (HGB, rs=7). Metrics: AUROC + TPR@FPR=1e-3 with Wilson CIs.
- Repro: `scripts/exp_d5.py` (main table), `scripts/probe_d5_tail.py` (stability).

## The tail hypothesis: confirmed as a phenomenon

DCT raw HGB-prob tail on B (n=500 human / 1000 AI):

| | p50 | p99 | p99.9 |
|---|---|---|---|
| human | 0.608 | 0.961 | 0.984 |
| AI | 0.786 | — | **share above human p99.9: 0.000** |

DCT's top tail is 100% human-occupied, and its raw probs run hot (human median 0.608
for a 0.655-AUROC detector). But every calibration here is monotone: none can put AI
mass into a corner of DCT-space where no AI ranks. Standalone DCT TPR@1e-3 = 0.010
is calibration-invariant. The interesting question is only what DCT's *spacing* does
to the meta-learner.

## Stack comparison (C bucket, cached base scores)

| condition | meta | AUROC | TPR@1e-3 [95% CI] |
|---|---|---|---|
| 3det raw | logreg | 0.925 | 0.076 [0.061, 0.094] |
| 3det raw | hgb | 0.903 | 0.175 [0.153, 0.200] |
| 4det raw | logreg | 0.928 | 0.026 [0.018, 0.038] |
| 4det raw | hgb | 0.920 | 0.234 [0.209, 0.261] |
| 4det platt(dct) (a) | logreg | 0.928 | 0.022 [0.015, 0.033] |
| 4det platt(dct) (a) | hgb | 0.920 | 0.234 [0.209, 0.261] |
| 4det isotonic(dct) (b) | logreg | 0.928 | 0.026 [0.018, 0.038] |
| **4det isotonic(dct) (b)** | **hgb** | **0.915** | **0.249 [0.223, 0.277]** |
| 4det quantile(all) (c) | logreg | 0.928 | 0.017 [0.011, 0.027] |
| 4det quantile(all) (c) | hgb | 0.920 | 0.234 [0.209, 0.261] |
| 4det logit(dct) (d) | logreg | 0.928 | 0.043 [0.032, 0.057] |
| 4det logit(dct) (d) | hgb | 0.920 | 0.234 [0.209, 0.261] |
| 4det platt(all) (bonus) | logreg | 0.925 | 0.033 [0.024, 0.046] |
| 4det platt(all) (bonus) | hgb | 0.920 | 0.234 [0.209, 0.261] |

Success bar (beat 0.170 TPR, hold ~0.896 AUROC): **cleared** by every 4-det hgb row.
Best tail: isotonic(dct)/hgb 0.249, AUROC 0.915 — +0.015 over raw, CIs overlap, so
treat as directional, not proven. It is also the only variant that moves the hgb
corner at all: platt/quantile/logit are strictly monotone, and HGB's quantile binner
makes the tree *exactly* invariant to them (bit-identical predictions). Isotonic
escapes invariance only because its fitted step function creates ties.

## Where the 0.155 went (probe_d5_tail.py)

- Re-running `stack_detectors.py` (recomputes base scores with this branch's
  0b723bf-era code) reproduces hgb = 0.155 exactly. The official cache gives 0.234.
  Same DCT, same buckets, same meta — the only swapped ingredient is the base-score
  provenance: cached qgram12 (current main code) has AUROC 0.893 / TPR 0.038 vs
  0.886 / 0.056 recomputed from 0b723bf-era code. **The dilution was qgram12 code
  drift interacting with DCT, not DCT miscalibration.** Fleet: treat pre-cache stack
  numbers as stale.
- The hgb meta's corner is *not* noise: column permutations and seeds {0,1,2,7,42}
  give bit-identical 0.234 (no early stopping at n=1500 → deterministic).
- logreg's linear tail is hopeless in the 1e-3 corner no matter the calibration
  (0.017–0.076); human outliers dominate a weighted sum. For low-FPR stacking,
  meta choice (hgb) ≫ calibration.

## Recommendation

Ship the 4-detector stack with hgb meta. Raw DCT already lifts the corner
(0.175 → 0.234); isotonic(dct) on B is a free, protocol-clean option that nudges to
0.249 at a −0.005 AUROC cost (0.915, still above the 0.896 bar). Don't stack raw
probabilities with a linear meta if you care about FPR ≤ 1e-3.
