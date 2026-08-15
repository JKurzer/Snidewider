# F3 — q-gram variants (worktree f3-qgram-variants)

Protocol: RAID dev fold, 2000 human + 4000 AI (rs=17), 50/50 source-disjoint
split (rs=23). Features cached in `data/derived/f3_features.parquet` (pure
functions; cache is convenience). Repro: `scripts/exp_f3.py`.

**Baseline reproduction check:** baseline9 full = logreg 0.898 / hgb 0.925 —
matches the fleet reference exactly. Pipeline is faithful.

##  Fleet-wide bug found: `metrics.auroc` tie handling

`auroc` sorted `(score, label)` tuples — among tied scores ALL humans rank
below ALL AI, fabricating separation for discrete features. `q5_count_p50`
scored **AUROC 0.996** and was pure artifact (both-class median = 1.0).
Fixed in this worktree (average ranks for ties, `tests/test_metrics.py`).
**Main checkout's `metrics.py` still has the bug — propagate the fix.**
Continuous features (series stats, probabilities) are unaffected.

## q sweep — per-feature AUROC (test half, tie-corrected)

Profile stats (flat in q; q=3 defaults were fine):

| feature         | q=2   | q=3   | q=4   | q=5   |
|-----------------|-------|-------|-------|-------|
| total           | 0.456 | 0.456 | 0.456 | 0.456 |  ← length proxy, mild
| distinct_ratio  | 0.515 | 0.521 | 0.528 | 0.522 |
| repeat_frac     | 0.475 | 0.481 | 0.479 | 0.487 |
| max_share       | 0.523 | 0.532 | 0.528 | 0.499 |
| entropy         | 0.377 | 0.393 | 0.398 | 0.398 |  ← inverted (AI lower), sep ~0.12

Collision spectrum (all ~0.5 = dead, post tie-fix): count_p50 0.503–0.520,
count_p90 0.471–0.495, count_p99 ~0.47, top10_share 0.549–0.564 (best).
Rare-event note: P(q5_count_p50 > 1) = 4.1% AI vs 0.4% human — tiny but
10×; HGB can use it, logreg can't.

## Change series — metric variants (test-half AUROC, sep = |x−0.5|)

| feature          | full  | matched | verdict |
|------------------|-------|---------|---------|
| mid_qgram_mean   | 0.117 | 0.147   | **strongest single feature (sep 0.38)** |
| mid_ck2_mean     | 0.230 | 0.277   | baseline |
| mid_bag_mean     | 0.226 | 0.279   | ≈ ck2 — bag adds NOTHING |
| mid_ck2_stdev    | 0.713 | 0.691   | |
| mid_qgram_stdev  | 0.712 | 0.702   | ties ck2 |
| mid_bag_stdev    | 0.668 | 0.667   | worse |
| short_*          | ~0.43–0.52 | ~0.44–0.52 | weak everywhere |

qgram metric on midrange windows: real win over ck2 (sep 0.383 vs 0.270),
holds under length matching. Bag distance: skip.

## Combined models (AUROC | TPR@1e-3 [Wilson])

Full subset (length bias intact; series features need ≥350 tokens → only
1131/6000 docs usable — THAT is the "biased subset"):

| set                    | logreg                | hgb                   |
|------------------------|-----------------------|-----------------------|
| baseline9              | 0.898 \| 0.202        | 0.925 \| 0.261        |
| baseline+qgram-series  | 0.896 \| 0.223        | 0.925 \| 0.287        |
| series-only            | 0.901 \| 0.202        | 0.902 \| 0.274        |
| **recommended (12)**   | **0.909 \| 0.253**    | **0.940 \| 0.316**    |
| profile-exp (all 6000) | 0.633 \| 0.053        | 0.817 \| 0.142        |

Length-matched subset (length-as-feature AUROC 0.484 ≈ chance; 729 usable):

| set                    | logreg                | hgb                   |
|------------------------|-----------------------|-----------------------|
| baseline9              | 0.898 \| 0.115        | 0.940 \| 0.197        |
| baseline+qgram-series  | 0.903 \| 0.120        | 0.942 \| 0.273        |
| **recommended (12)**   | **0.906 \| 0.137**    | **0.944 \| 0.208**    |
| profile-exp (all 4000) | 0.589 \| 0.057        | 0.743 \| 0.130        |

## Recommended qgram-family set (12 features)

`short_ck2_mean/stdev`, `mid_ck2_mean/stdev`, **`mid_qgram_mean/stdev`**,
`q2_entropy`, `q3_entropy`, `q3_distinct_ratio`, `q3_repeat_frac`,
`q3_max_share`, `q5_top10_share`. Dropped `qgram_total` (length leak).

- vs baseline9: hgb AUROC 0.925→0.940 (full), 0.940→0.944 (matched);
  TPR@1e-3 0.261→0.316 (full). Gain survives length matching → not a
  length artifact.
- profile-exp (q-sweep stats + spectrum, no series) covers ALL docs at
  hgb 0.817 — a coverage/fallback option for short docs, not a replacement.
- Coverage caveat stands: series features see only 19% of docs. A short-doc
  route (profile-exp) + long-doc route (recommended) is the natural next step.
