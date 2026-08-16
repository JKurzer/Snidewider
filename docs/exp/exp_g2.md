# exp_g2 — burst-stage sweep of the dct_run CK2 series (G2, one-sided gate hunt)

**Question:** with the dct_run encoder fixed (RUN=8, k=2, 4-bit), does any
window/step/statistic combination of the CK2 burst series produce a ONE-SIDED
gate — AI mass where no human lives (zero-FPR TPR), or a threshold that
catches all AI cheaply (min-FPR at full TPR)?

**Script:** `scripts/exp_g2.py` — run from the main checkout cwd so `data/`
resolves (code comes from this worktree via PYTHONPATH):

```
cd C:\Users\poly\ai-text-detection
set PYTHONPATH=C:\Users\poly\ai-text-detection-g2-burststage\src&& C:\Users\poly\ai-text-detection\.venv\Scripts\python C:\Users\poly\ai-text-detection-g2-burststage\scripts\exp_g2.py
```

Data: `raid_splits.parquet`, dev fold only. Sweep: 1000 human + 1000 AI
(random_state=51). Confirm: 2000 human + 4000 AI (random_state=52; dev has
exactly 2000 humans, so the confirm human side is the full dev human set).
Both score orientations gated. Full per-config dump (gitignored):
`data/derived/exp_g2_results.json`.

## What is swept

| axis | values |
|---|---|
| window (symbols, 1 byte = 1 symbol) | 16, 32, 64, 128 |
| layout | adj (gap=0, step=W), half (gap=-W/2, step=W/2), gapw (gap=W, step=2W), rand (32 pairs, min_gap=W) |
| series stat | mean, stdev, min, max, p01, p05, p25, p95, p99, iqr |
| reference | incumbent production path (latin-1 decode -> utf-8 byte windows, w32) |

Note: the sweep runs CK2 over the RAW symbol bytes, so "window" means exactly
that many symbols. The incumbent path re-encodes latin-1 -> utf-8, so bytes
>= 0x80 (most dct_run symbols: high nibble is quantized mean length) take 2
bytes and the effective window becomes data-dependent mush. Incumbent rows are
included for head-to-head.

## Coverage reality (the binding constraint)

dct_run emits 1 symbol per 8 tokens; dev docs are short:
symbol-stream length p10=19, **median=29**, p90=70 (first 200 sampled docs).
Minimum stream needed: adj/w16 = 32, half/w16 = 24, rand/w16 = 48 symbols.

| series key | coverage (min of classes) |
|---|---|
| half/w16 | **0.60** (hu 0.73 / ai 0.60) |
| adj/w16 | 0.40 |
| incumbent/w32 | 0.26 |
| rand/w16 | 0.11 |
| everything w>=32 | <= 0.11 (most 0.00) |

No config in the prescribed grid reaches the 80% coverage bar. The constraint
is the fixed encoder vs. doc lengths, not the series stage.

## Sweep highlights (1000 hu + 1000 ai, seed 51)

zero-FPR TPR = fraction of valid AI docs beyond the human extreme (lo = AI
docs CALMER than the calmest human; hi = wilder than the wildest human).
minFPR-at-full-TPR was **1.000 for every config** — the no-FN direction is
dead on arrival; human mass covers the whole AI range.

| config | cov | AUROC | zfTPR lo [CI] | zfTPR hi [CI] |
|---|---|---|---|---|
| rand/w16/mean | 0.11 | 0.891 (lo) | **29.9% [22.1, 39.2]** | 0 |
| rand/w16/max, p99, p95 | 0.11 | 0.69 | 26-27% | 0 |
| half/w32/mean | 0.11 | 0.59 | 25.2% [18.0, 34.2] | 5.6% [2.6, 11.7] |
| adj/w16/p05 | 0.40 | 0.51 | **12.3% [9.5, 15.6]** | 0 |
| adj/w16/p25 / mean / min | 0.40 | 0.52-0.55 | 11.6-12.0% | 0 |
| half/w16/p25 | 0.60 | 0.55 | 8.1% [~6, ~11] | 0 |
| half/w16/p05, p01, min | 0.60 | 0.54 | 8.0% | 0 |
| half/w16/stdev | 0.60 | 0.58 (hi) | 0 | 3.6% (a few AI docs wilder than any human) |
| incumbent/w32/mean | 0.26 | 0.58 (lo) | 9.4% | 3.5% |
| incumbent/w32/stdev (= current prod feature) | 0.26 | 0.725 (lo) | **0.0%** | 0.3% |

The incumbent's chosen stat (stdev) is the one stat with NO lo-tail — the
one-sided signal lives in the mean/quantile floor, not the spread.

## Confirmation (seed 52, 2000 hu + 4000 ai)

Protocol top-3 (ranked by sweep zfTPR), plus a targeted confirm of the
coverage leaders:

| config | cov | zfTPR lo [CI] |
|---|---|---|
| rand/w16/mean | 0.11 | **19.1% [15.7, 23.2]** (sweep 29.9% — shrank, solid) |
| rand/w16/p99 / max | 0.11 | 16.5-17.3% |
| adj/w16/p25 | 0.39 | **8.5% [7.3, 9.9]** |
| adj/w16/mean / p05 / p01 / min | 0.39 | 7.8-8.1% |
| half/w16/mean | 0.61 | **5.3% [4.5, 6.3]** |
| half/w16/p25 / p05 | 0.61 | 4.6-4.8% |

Tails replicate with tight CIs at both coverage tiers.

## Length-confound control (ablation)

Is the lo-tail just "long docs give more series points"? No:
valid AI streams are if anything SHORTER than valid human streams (all valid
AI < 64 symbols; humans extend past 96). Within length-matched bands
(seed-51 sample, adj/w16/mean lo):

| stream length | hu / ai n | zfTPR lo [CI] |
|---|---|---|
| [32, 48) | 232 / 342 | 7.9% [5.5, 11.2] |
| [48, 64) | 68 / 107 | 27.1% [19.6, 36.2] |

The tail survives matching and grows with length — a per-window-content
effect (sustained rhythmic calm), not a series-length artifact. Per-class
Spearman corr(stat, length): +0.04 human, -0.17 AI — weak.

## Verdict

**One-sided signal: FOUND and confirmed.** ~5-9% of AI docs (at usable
coverage) sit below the human minimum of windowed-CK2 rhythm distance —
zero observed false positives, CIs comfortably clear of 0, replicated on the
seed-52 draw, not a length artifact. The detectable thing is *sustained
rhythmic calm*: AI docs whose token-length rhythm repeats itself more than
any human's.

**Gate success as specified: NOT met.** Coverage > 80% is unreachable in the
prescribed grid — ceiling is 0.60 (half/w16), because RUN=8 yields a median
29-symbol stream and the smallest swept window still needs 24-32. The no-FN
probe (minFPR at full TPR) is 1.000 everywhere.

**Hand-off recommendation:** the lo-tail is worth harvesting. To lift
coverage past 80%, the next gate owner should shrink the ENCODER run
(RUN=4 doubles stream length; w16 half would cover ~95%+) or allow window=8
(adj needs only 16 symbols, ~90% coverage) — both outside G2's mandate
(encoder + window set were fixed). As-is, half/w16/p05-lo or adj/w16/p25-lo
are viable cascade pre-gates firing on ~3% of ALL AI docs at zero FP.

Tests: 78 passed in this worktree (pytest run from main cwd for data paths).
