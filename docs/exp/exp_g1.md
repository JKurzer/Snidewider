# G1 — one-sided gate hunt over the dct_run encoder family

**Headline: one-sidedness EXISTS in the dct_run family, but only as a partial
no-FP gate — and it lives in the EXTREMES, not the carrier. Best general gate:
RUN=4, levels=8, k=2, statistic = series min, low orientation: 8.9% [7.7, 10.3]
of covered AI docs sit strictly below the human minimum at measured FPR 0
(788 covered humans), coverage 45.1% AI / 39.4% human → ≈4% of ALL AI docs
auto-flagged. Not a length artifact (pocket persists in the 1-pair bucket:
6.4% [5.3, 7.9]). The no-FN side is dead: min-FPR@full-TPR ≈ 1.0 at any real
coverage. The shipped stdev carrier is never one-sided (noFP ≤ 0.009
everywhere) — the strategy note's guess was right: min/max is the habitat.**

## Protocol (RULES #3/#4)

- Data: `data/derived/raid_splits.parquet`, **dev fold only** (holdout
  untouched). Sweep: 1000 human + 1000 AI, `random_state=51`. Rerun of the
  selected configs: 2000 human (= all dev humans) + 4000 AI,
  `random_state=52`.
- Sweep grid: RUN ∈ {4, 8, 16} tokens/run × quantization levels/coeff
  ∈ {2, 4, 8, 16} × k coeffs ∈ {1, 2, 3} = 36 encoder configs.
- Burst stage fixed at the shipped carrier path: stepped series over the
  latin-1 view of the symbol bytes, ck2, window=32, gap=0. Statistics:
  stdev (the carrier), min, max (the extreme habitat).
- Probes per config × stat × orientation (both checked): `zero_fpr_tpr`
  (no-FP: AI mass strictly above the human max) and `min_fpr_at_full_tpr`
  (no-FN: human mass above the AI min), Wilson CIs, plus coverage (fraction
  of docs with a finite value, per class). NaN docs are excluded from the
  probe and counted as uncovered — a gate that can't fire on short docs is
  a sniper, not a gate.
- Repro: `scripts/exp_g1.py` (sweep + rerun), `scripts/exp_g1_lenbias.py`
  (length-confound probe). Full probe rows:
  `data/derived/exp_g1_sweep.csv`, `exp_g1_top3.csv` (worktree, gitignored).

## Encoder parameterization (and sanity)

DCT-II of the clipped token-length sequence per RUN-token run; c0 quantized
over mean-length range [0, 8), cj>0 over [-8, 8) — the shipped map's ranges,
with c2 inheriting c1's range (a choice, flagged). Symbols are mixed-radix
(q0·L^(k-1) + …), 1 byte if L^k ≤ 256 else 2 bytes big-endian. Sanity:
the (8, 16, 2) config reproduces `shape.dct_run_map` with byte-agreement
1.0000 (2043 symbols, 50 docs) — the sweep grid contains the shipped point.

## Coverage is a byte-budget phenomenon (read before the tables)

A doc scores only if its symbol stream yields ≥ 64 carrier bytes (one
window pair). The carrier's latin-1→utf-8 round-trip doubles bytes for
symbols ≥ 0x80, so coverage depends on RUN **and on whether the packing
pushes symbols into the high byte range** (e.g. R4/L16/k2 symbols start at
~0x80 → ~2 bytes/symbol → 74% AI coverage; R4/L16/k1 stays < 0x80 → 45%).
RUN=16 halves symbol counts → ~3% AI coverage: a long-doc sniper by
construction. Docs here are short (median 235 tokens): even covered docs
carry only 1–2 window pairs.

## Sweep leaderboards (1000 H + 1000 AI; best probe per config shown)

Board A — no-FP gate candidates (zero-FPR TPR desc):

| cfg | stat | orient | cov ai/hu | noFP TPR [CI] | noFN FPR [CI] |
|---|---|---|---|---|---|
| R16 L8 k3 | min | low | .029/.116 | **0.483 [0.314, 0.656]** | 0.974 [0.927, 0.991] |
| R16 L16 k3 | max | low | .037/.148 | 0.324 [0.196, 0.485] | 1.000 |
| R4 L8 k2 | min | low | .449/.400 | **0.111 [0.085, 0.144]** | 1.000 |
| R4 L4 k3 | min | low | .449/.400 | 0.109 [0.084, 0.141] | 0.998 |
| R8 L16 k3 | min | low | .565/.647 | 0.101 [0.079, 0.128] | 1.000 |
| R4 L16 k2 | min | low | **.740/.893** | 0.100 [0.080, 0.124] | 1.000 |
| R4 L16 k1 | min | low | .449/.400 | 0.094 [0.070, 0.124] | 0.998 |
| R8 L16 k2 | max | low | .339/.256 | 0.091 [0.065, 0.127] | 1.000 |
| R4 L4 k2 | min | low | .449/.400 | 0.089 [0.066, 0.119] | 1.000 |
| R8 L8 k3 | min | low | .530/.575 | 0.075 [0.056, 0.101] | 0.998 |

Every no-FP pocket is a **min/low** (or max/low) probe: AI docs whose rhythm
has a *suspiciously uniform* stretch no human matches. stdev never appears.

Board B — no-FN candidates (min-FPR asc): the champion is R16 L8 k3 stdev-low
at FPR 0.509 [0.419, 0.598] — on 2.9% AI coverage. Everything with real
coverage sits at FPR ≈ 1.0. **No no-FN gate exists in this family.**

Selection rule for the rerun (coverage-aware): top-3 of board A restricted to
cov_ai ≥ 0.40, plus the board-A sniper #1, plus the shipped carrier (8,16,2)
as baseline. Board B earned no slot (champion has no real coverage — that
absence is the no-FN finding).

## Rerun (2000 H + 4000 AI; low orientation = the habitat)

| cfg | stat | cov ai/hu | noFP TPR [CI] | noFN FPR |
|---|---|---|---|---|
| **R4 L8 k2** | **min** | .451/.394 | **0.089 [0.077, 0.103]** | 1.000 |
| R4 L8 k2 | max | .451/.394 | 0.081 [0.069, 0.094] | 1.000 |
| R4 L8 k2 | stdev | .451/.394 | 0.000 [0.000, 0.002] | 1.000 |
| R4 L4 k3 | min | .451/.394 | 0.077 [0.066, 0.090] | 1.000 |
| R8 L16 k3 | min | .565/.628 | 0.034 [0.027, 0.042] | 1.000 |
| R16 L8 k3 (sniper) | max | .027/.115 | **0.533 [0.439, 0.624]** | 0.970 |
| R16 L8 k3 (sniper) | min | .027/.115 | 0.243 [0.172, 0.332] | 1.000 |
| R8 L16 k2 (carrier) | min | .336/.258 | 0.046 [0.036, 0.059] | 1.000 |
| R8 L16 k2 (carrier) | stdev | .336/.258 | 0.000 [-, 0.003] | 1.000 |

R4 L8 k2 min-low replicates (sweep 0.111 → rerun 0.089; mild attenuation =
selection bias + a bar set by 2× more humans, still solidly non-zero). Note
the shipped carrier has the pocket too (min-low 0.046) — earlier passes
missed it because they only ever measured stdev and TPR@1e-3. Also worth
flagging: min-low AUROC is 0.434 — *below chance* — yet the pocket exists.
Rank statistics are blind here; one-sided probes are not optional.

Deployed as an auto-flag gate (uncovered docs fall through to the panel and
cannot false-flag): R4 L8 k2 min-low flags ≈ 4.0% of all AI (0.451 × 0.089)
at measured FPR 0/2000 humans (Wilson upper ≈ 0.19%).

## Length-confound probe (`exp_g1_lenbias.py`, rerun sample)

min-of-series mechanically shrinks with more pairs; covered AI docs could
just be longer. Bucketed by pair count, probing within buckets:

| pairs | n_ai | n_hu | human min | AI below bar | TPR [CI] |
|---|---|---|---|---|---|
| 1 | 1382 | 459 | 0.329 | 89 | 0.064 [0.053, 0.079] |
| 2 | 423 | 139 | 0.289 | 84 | 0.199 [0.163, 0.239] |

The pocket survives at **one pair** — a single adjacent-window CK2 value,
where no length bias is possible. The uniformity pocket is rhythm, not
length. (The 2-pair bucket is even stronger, but its bar rests on only 139
humans.)

## Verdict

- **Does one-sidedness exist here? YES — one side of it.** A no-FP pocket is
  real and replicated: AI docs produce suspiciously uniform rhythm stretches
  that no covered human doc produces. It is a *partial* gate (single-digit %
  of AI), not a panel replacement.
- **No-FN gate: does not exist.** Catching every covered AI doc costs ≈ all
  humans at any usable coverage; the AI low tail always overlaps humanity.
- The carrier stdev is a bulk statistic and stays two-sided; extremes
  (min/max, low orientation) are the one-sided habitat, exactly as
  `docs/dct-run-family.md` predicted.
- Best config: **RUN=4, levels=8, k=2, min, low**. Sniper alternate:
  R16 L8 k3 max-low (53% of covered AI, but covers 2.7%).

## Caveats / follow-ups

- Dev-only, and the winner was selected on humans overlapping the rerun
  (rerun used all 2000 dev humans, half untouched by the sweep sample; AI
  rerun sample is fresh modulo ~59 expected overlapping rows). Holdout
  confirmation: once, clean, when the fleet schedules it.
- R4 L16 k2 is the coverage play: sweep-only evidence (cov 74% AI × noFP
  0.100 → ≈7.4% of all AI flagged) — it missed the rerun cut by 0.001.
  First follow-up candidate.
- Quantization ranges for c2 were assumed (= c1's); coverage is hostage to
  the latin-1→utf-8 byte-expansion quirk — a byte-true carrier would
  reshuffle coverage (and is arguably the honest burst input).
