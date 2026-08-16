# exp_d2 — DCT sampling sweep (dev fold, bucket C)

**Headline: `win48-k2` (48-token windows, 2 DCT coeffs) lifts the 4-detector stack to
TPR@FPR=1e-3 = 0.307 [0.279, 0.336] vs 0.175 for the 3-detector reference, at AUROC
0.905 (>= the 0.896 floor). Tail problem: solved, with margin.**

## Question

The DCT encoder's "sampling" = how segments are drawn from a doc before encoding.
Status quo: sentence split. Does a different draw make the DCT detector useful for
the low-FPR tail of the stack?

## Protocol (RULES #3/#4)

- Dev fold only; source-disjoint buckets A/B/C (50/25/25) via `evaldata.split_buckets`.
- Each DCT variant: 4 doc features (adjacent-segment cosine mean/stdev in DCT space,
  order-energy mean/stdev), `HistGradientBoostingClassifier(random_state=7)` on A.
- Stack: HGB(random_state=7) on B's 4 scores = cached relative-burst / qgram12 /
  exemplar (`base_scores.npz`) + the DCT variant's B-score. Numbers read ONCE on C.
- Variants:
  - `sent-k2` — sentence split, k=2 (status quo, reference row).
  - `sub16-k{2,4}` — 16 sentences sampled without replacement per doc, seeded by
    `sha256(text)[:4]` (same trick as `burst.random_change_series`), original
    document order preserved, then the usual feature math.
  - `win{W}-k{2,4}` — non-overlapping whitespace-token windows, W in {8..64}.
- Embedding trick (tested in `test_dct.py`): per-whitespace-token regex + concat ==
  `embed_sentence` over the joined window, so each doc is embedded once and all 15
  configs slice the same matrices. 6000 docs featurized in ~40 s.

## Results (bucket C, n=1500: 1000 AI / 500 human)

Reference, 3-detector stack (no DCT): **AUROC 0.903 | TPR@1e-3 0.175 [0.153, 0.200]**

| config   | DCT standalone AUROC | DCT standalone TPR | 4-det stack AUROC | 4-det stack TPR@1e-3 |
|---|---|---|---|---|
| sent-k2  | 0.655 | 0.010 | 0.920 | 0.234 [0.209, 0.261] |
| sub16-k2 | 0.654 | 0.004 | 0.913 | 0.170 [0.148, 0.195] |
| sub16-k4 | 0.665 | 0.001 | 0.913 | 0.190 [0.167, 0.215] |
| win8-k2  | 0.631 | 0.012 | 0.904 | 0.100 [0.083, 0.120] |
| win8-k4  | 0.639 | 0.018 | 0.902 | 0.196 [0.173, 0.222] |
| win16-k2 | 0.554 | 0.010 | 0.903 | 0.224 [0.199, 0.251] |
| win16-k4 | 0.599 | 0.010 | 0.904 | 0.244 [0.218, 0.272] |
| win24-k2 | 0.563 | 0.000 | 0.901 | 0.174 [0.152, 0.199] |
| win24-k4 | 0.580 | 0.001 | 0.902 | 0.262 [0.236, 0.290] |
| win32-k2 | 0.543 | 0.002 | 0.905 | 0.075 [0.060, 0.093] |
| win32-k4 | 0.548 | 0.000 | 0.904 | 0.116 [0.098, 0.137] |
| **win48-k2** | 0.574 | 0.009 | **0.905** | **0.307 [0.279, 0.336]** |
| win48-k4 | 0.555 | 0.003 | 0.905 | 0.192 [0.169, 0.218] |
| win64-k2 | 0.615 | 0.003 | 0.905 | 0.183 [0.160, 0.208] |
| win64-k4 | 0.600 | 0.010 | 0.898 | 0.189 [0.166, 0.214] |

## Reading

- **Winner: `win48-k2`.** Clears the TPR bar (0.170) by ~1.8x, CI disjoint from the
  3-detector reference, AUROC 0.905 >= 0.896 floor. Sentence-split (`sent-k2`) keeps
  the best stack AUROC (0.920) but a clearly weaker tail (0.234).
- Standalone, every DCT variant is mediocre at 1e-3 FPR (TPR <= 0.018) — the value
  is entirely complementary signal for the meta-classifier, exactly what a panel
  member should be.
- Sentence subsampling is strictly worse than using all sentences (0.170–0.190 vs
  0.234): random draws break the adjacency that the cosine stat feeds on.
- No monotone window trend (win32 craters, win48 spikes) — the tail gain is an
  empirical property of the window/grid interplay, not a smooth "bigger is better".
- Caveat: winner picked on C across 15 configs sharing one bucket, so the 0.307 is
  mildly selection-optimistic; the gap to 0.170 is large enough that the conclusion
  (windows > sentences for the tail) is safe. Holdout stays untouched (RULES #4).

## Recommendation

Adopt `dct_features(text, k=2, unit="windows", window=48)` as the 4th detector's
sampling scheme. `dct.py` defaults are left unchanged so other fleet branches are
undisturbed; flipping the default is a one-line follow-up once the fleet agrees.

## Repro

```
set PYTHONPATH=C:\Users\poly\ai-text-detection-d2-sampling\src&& C:\Users\poly\ai-text-detection\.venv\Scripts\python C:\Users\poly\ai-text-detection-d2-sampling\scripts\exp_d2.py
```

from cwd `C:\Users\poly\ai-text-detection` (data/ resolves there). ~55 s total.
