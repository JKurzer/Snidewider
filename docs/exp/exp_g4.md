# G4 — per-token scalar swap in the DCT-run encoder: one-sided gate hunt

Worktree `ai-text-detection-g4-scalars`, branch `exp/g4-scalars`.
Repro: `scripts/exp_g4.py` (run from main-repo cwd, PYTHONPATH=this worktree's src).
Code: `src/ai_text_detection/scalar_maps.py` (shape.py fork), tests in
`tests/test_scalar_maps.py` (83 passed).

**Hypothesis (LLM-blindness):** character-composition side channels are
under-trained in LLM output, so per-token scalars other than length may
expose one-sided extremes (AI-only regions above the human max, or
human-only regions below the AI min).

**Machinery:** identical run(8)/DCT-II(K=2)/quantize(nibbles)/burst(stepped,
gap 0, ck2) pipeline as `shape.dct_run_map`; ONLY the per-token scalar
changes. Stats per scalar: mean/stdev/min/max/p05/p95 of the stepped series
=> 30 gate candidates, both orientations checked.

## Scalars (scaled into the length-like [0,15] nibble range)

| scalar | definition |
|--------|------------|
| len    | min(len(tok), 15) — control |
| vowel  | min(15, 6 * vowels / max(1, consonants)) |
| ccclass| 2 * code; precedence digit(2) > punct/`'`(4) > ALLCAPS(3) > Capitalized(1) > lower(0) > mixed(5) |
| caps   | 15 * uppercase fraction |
| shape2 | 3 * (2*(len>3) + (vowels>consonants)) — 2-bit coarse class |

## Deviation from spec: burst window 32 -> 6 (measured, not guessed)

Spec'd w32 needs >=512 tokens/doc for one comparison. RAID-dev token
percentiles p10/p50/p90 = 101/240/387 (humans skew longer):

| window | tokens needed | cov human | cov AI |
|--------|---------------|-----------|--------|
| 32 (spec) | 512 | 0.115 | **0.000** |
| 8      | 128           | 0.984 | 0.811 |
| 6 (used) | 96          | 0.996 | 0.908 |

w32 is a dead config on RAID dev (zero AI coverage); w6 is the largest
window holding >=80% on both classes with margin. In-run coverage achieved:
0.91 AI / 0.99 human — gate-eligible.

Two more fixes vs shape.py, documented for the fleet: (1) burst windows here
align to SYMBOLS — shape.py's latin-1/utf-8 roundtrip through
`burst_features` misaligns windows whenever the hi-nibble exceeds 7;
(2) confirm sets are sweep-disjoint (below).

## Protocol (dev fold ONLY, RULES #3/#4)

- Sweep: 1000 AI + 1000 human (rs=51). Rank by zero-FPR TPR, tie-break
  lowest FPR@fullTPR; eligibility = coverage >=0.80 both classes.
- Confirm: top-3 on 4000 AI (rs=52, sweep rows excluded) + the 1000 humans
  the sweep never saw. Reference line: bar over ALL 2000 dev humans
  (stricter bar but selection-contaminated — humans fully overlap sweep).
- Metrics: `zero_fpr_tpr`, `min_fpr_at_full_tpr` (Wilson CIs),
  direction-corrected AUROC, coverage. Success = meaningful zero-FPR TPR or
  clearly-sub-0.3 FPR@fullTPR at >=80% coverage.

## Sweep (all 30 eligible features, ranked; cov 0.91/0.99 throughout)

| feature | orient | AUROC | zeroFPR TPR [CI] | FPR@fullTPR |
|---------|--------|-------|------------------|-------------|
| shape2_min  | - | 0.539 | **0.021 [0.013, 0.032]** | 1.000 |
| shape2_p05  | - | 0.543 | 0.020 [0.013, 0.031] | 1.000 |
| vowel_min   | - | 0.516 | 0.016 [0.010, 0.027] | 1.000 |
| vowel_p05   | - | 0.526 | 0.016 [0.010, 0.027] | 1.000 |
| len_mean    | - | 0.515 | 0.012 [0.007, 0.022] | 1.000 |
| shape2_mean | - | 0.564 | 0.011 [0.006, 0.020] | 1.000 |
| vowel_mean  | - | 0.548 | 0.010 [0.005, 0.019] | 1.000 |
| vowel_max/p95, shape2_max/p95 | - | 0.535-0.554 | 0.002 | 1.000 |
| len_max/p95, stdev, min(+), p05(+) | mixed | 0.513-0.551 | <=0.001 | 1.000 |
| vowel_stdev | - | 0.522 | 0.000 | 1.000 |
| ccclass_* (6) | - | 0.504-0.581 | 0.000 | 1.000 |
| caps_* (6)    | - | 0.524-0.563 | 0.000 | 1.000 |
| shape2_stdev  | - | 0.507 | 0.000 | 1.000 |

Every FPR@fullTPR = 1.000 [0.996, 1.000]: the AI minimum always sits at the
bottom of the human range — no no-FN one-sidedness whatsoever.

## Confirm (4000 AI + 1000 unseen humans) — sweep tails evaporate

| feature | orient | AUROC | zeroFPR TPR [CI] | FPR@fullTPR |
|---------|--------|-------|------------------|-------------|
| shape2_min | - | 0.533 | **0.000 [0.000, 0.001]** | 1.000 |
| shape2_p05 | - | 0.545 | 0.000 [0.000, 0.001] | 1.000 |
| vowel_min  | - | 0.528 | 0.000 [0.000, 0.001] | 1.000 |

Reference bar over all 2000 humans: identical (0.000 everywhere).

## Verdict: NO HIT

- No scalar — including the length control — shows one-sided tail behavior
  on RAID dev. The best sweep tail (2.1% of AI beyond the human max) did
  not survive confirmation (0.0%, CI hi 0.1%): textbook selection noise at
  n=1000.
- Composition scalars add nothing over the control: ccclass/caps AUROCs
  (0.50-0.58) match len, and their zero-FPR tails are exactly zero.
- Direction note: wherever any separation exists it is `-` (AI distances
  LOWER = locally more uniform) — consistent with the burstiness
  literature, but at AUROC <= 0.58 it is gate-useless.
- The DCT-run rhythm family (any scalar) at RAID doc lengths (~240 tokens
  median, ~30 symbols, ~4 series points/doc) is too starved for tail gates.
  If the fleet revisits: longer docs or a denser symbol stream (RUN<8)
  would raise series density; but with FPR@fullTPR pinned at 1.000 the
  no-FN route looks structurally dead for this family.
