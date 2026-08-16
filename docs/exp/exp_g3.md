# exp_g3 — one-sided gates on SHORT docs (dct_run/skeleton family, short-doc knobs)

Fleet: G3. Worktree exp/g3-shortdocs. Script: `scripts/exp_g3.py` (run from the
main-repo cwd; see its docstring). Runtime ~80s.

## Question

Stock `dct_run` (RUN=8, burst window 32) dies on short docs: a 200-token doc
yields 25 symbols and window-32 burst never fires. Can the family be adapted
to docs ≤300 tokens (RUN ∈ {2,3,4}, window ∈ {4,8,16}, samples ≤16,
quantization ≤4 levels, plus skeleton), and does any adaptation show
ONE-SIDED behavior (zero-FPR flag gate / full-TPR pass gate)?

## Protocol

- Dev fold only. All 2000 dev humans + 4000 dev AI (`sample(random_state=52)`).
- Source-disjoint A/B/C via the fleet salt=41 source partition
  (evaldata.split_buckets' shuffle, minus the per-source AI cap so short
  cohorts keep real n's). **Selection on A+B; finalists verified once on C.**
- Cohorts: SHORT = ≤300 whitespace tokens (n=4258: 1539 hu / 2719 ai);
  TINY = 100–200 (n=1972: 689/1283); FULL = all 6000 for reference.
- Gate probes: `zero_fpr_tpr` + `min_fpr_at_full_tpr` with Wilson CIs; AUROC
  direction-corrected (orientation fixed on the same data being reported).
- COVERAGE = fraction of cohort docs with a finite gate score — headline.
- Machinery notes: burst series are computed bytes-native (burst.py's
  str→utf-8 path double-encodes dct symbols ≥0x80); stats are tiny-n tolerant
  (burst_features crashes on 1-element series); the samples=8 random series is
  the exact prefix of samples=16 (same draw order), computed once.

## Reference: the stock config is dead on shorts

`dct_r8_l16 / w32_s32` on A+B: coverage **0.000 on SHORT and TINY**, 0.031 on
FULL (the full sample is short-dominated). The family as shipped simply does
not exist below ~500 tokens.

## Adapted family: coverage is fixed, AUROC is thin

Coverage on A+B SHORT reaches 0.91–0.999 across the grid (w4/w8 near 1.0;
skeleton ≥0.996 everywhere). AUROCs, however, are 0.50–0.56 — moments of the
burst series barely separate on shorts. The one-sided tails are where the
signal lives:

**FLAG leaderboard (A+B, SHORT, coverage ≥0.80), top rows:**

| map | burst | stat | cov | auroc | dir | zero-FPR TPR [CI] | full-TPR FPR |
|---|---|---|---|---|---|---|---|
| dct_r2_l4 | w16_s16 | step_max | 0.970 | 0.549 | ai>lower | 0.0249 [0.0190, 0.0327] | 1.000 |
| dct_r2_l4 | w16_s8 | rand_mean | 0.914 | 0.533 | ai>lower | 0.0235 [0.0175, 0.0315] | 1.000 |
| dct_r3_l4 | w8_s16 | rand_mean | 0.958 | 0.531 | ai>lower | 0.0214 [0.0158, 0.0287] | 0.999 |
| dct_r2_l4 | w16_s16 | rand_min | 0.914 | 0.558 | ai>lower | 0.0202 [0.0147, 0.0278] | 1.000 |

**PASS gates: dead.** Every config's FPR at full TPR is 0.996–1.000 — the
distributions overlap completely at the human end. No pass-through gate
exists in this family on shorts.

## Union flag-gate (the candidate)

OR of the top-6 diverse zero-FPR stats (A+B auroc ≥0.52 filter, deduped by
(map, stat); components: dct_r2_l4 w16 step_max, dct_r2_l4/r3_l4/r2_l2
w8–w16 rand_mean, dct_r2_l4 w16 rand_min, dct_r3_l4 w8 rand_max — all
ai>lower, i.e. "suspiciously smooth rhythm"). Each component thresholded
strictly above the A+B SHORT human max of its oriented score → A+B FPR = 0
by construction; C is the honest test.

| bucket | cohort | TPR [CI] | FPR [CI] | coverage |
|---|---|---|---|---|
| A+B | SHORT | 0.0535 [0.0446, 0.0639] | 0.0000 [—, 0.0033] | 0.957 |
| **C** | **SHORT** | **0.0545 [0.0393, 0.0752]** | **0.0052 [0.0014, 0.0189]** | **0.950** |
| C | TINY | 0.0533 [0.0331, 0.0849] | 0.0111 [0.0030, 0.0394] | 1.000 |
| C | FULL | 0.0748 [0.0596, 0.0934] | 0.0040 [0.0011, 0.0145] | 0.967 |

The tail survives verification: ~1 in 18 AI shorts fires, at ~1/200 human
cost (C FPR is small but NOT zero — the A+B human max underestimates the
tail; reported honestly). Best individual gate on C SHORT:
`dct_r2_l4 w16 step_max`, zero-FPR TPR 0.0219 [0.0129, 0.0371], coverage 0.968.

## Caveats

- Direction flips at auroc ≈ 0.51 are coin flips: `dct_r3_l4 step_mean` and
  `skeleton w8 rand_stdev` flipped sign between A+B and C. Anything below
  ~0.52 AUROC here is noise; the union filter exists because of this.
- TINY cohort directions are less stable than SHORT (smaller n, shorter
  streams); trust SHORT numbers first.
- The zero-FPR property holds exactly only on tuning data; verified C FPR is
  0.4–1.1% depending on cohort.

## Verdict

**Partial success, flag-side only.** The adaptation works: coverage goes
0.000 → 0.95+ on SHORT (≥80% bar cleared with room). One-sided behavior is
REAL but THIN: a verified ~5.4% zero-nearly-zero-FPR auto-flag on AI shorts
(union gate), with the pass-through side nonexistent (full-TPR FPR ≈ 1.0).
The union gate is a usable high-precision tripwire feeding the panel on
shorts — it is not a standalone detector, and the family's center of mass
(AUROC ≤ 0.56) says most AI shorts remain the panel's problem.
