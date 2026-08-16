# dct_run_map — expansion into a feature family (+ the one-sided gate idea)

Status: NOTE, not an experiment. Do not build until scheduled.

## The expansion

`dct_run_map` (shape.py) currently maps each 8-token run -> DCT-II of token
lengths -> 2 quantized coefficients -> 1 byte. That is ONE point in a large
space, each axis of which is a feature-family generator:

- **scalar per token**: length (current) | vowel/consonant ratio | char-class
  pattern | capitalization | digit presence | syllable proxies
- **run length**: RUN=8 (current) | 4 | 16 | sentence-delimited runs
- **encoder**: DCT-II k=2 (current) | k=1..6 | plain run stats (no transform)
- **quantization**: 4 bits/coeff (current) | coarser/finer | more coefficients
  per symbol (pack 2-3 coeffs into a byte)
- **burst stage**: stepped (current) | random long-range | gap variants |
  window sizes | min/max statistics instead of mean/stdev — note: one-sided
  behavior would live in EXTREME statistics, which we have never computed

## The one-sided-gate strategy (Donk's framing)

If any member of this family can be tuned to **no false positives** (a hit
means certain AI — an auto-flag gate) or **no false negatives** (a miss means
certain human — a pass-through gate), then the rest of the classification
burden falls to the existing panel. One crisp side is worth more than another
0.01 of AUROC.

## Review of tuning passes to date (measured, no new compute)

- exp_shape (first pass): `dct_run_step_stdev` sep 0.727 solo; 8-feature
  shape set solo AUROC 0.860, solo TPR@1e-3 0.001.
- feature_report (C bucket): same feature, sep 0.764, coverage 0.428,
  TPR@1e-3 0.000 [0.000, 0.007], achieved FPR 0.0.
- 5-detector stack: +AUROC (0.949), -tail (0.116) — same dilution pattern as
  stock DCT.

### Is one-sidedness promising?

Evidence for: the direction is stable and coherent (AI rhythm steadier), and
stability across two independent passes suggests a real phenomenon, not
noise. A "suspiciously perfect rhythm" extreme (stdev ≈ 0) might be AI-only
territory — the no-FP-gate candidate.

Evidence against: at the only corner we've measured (FPR 1e-3), the tails
overlap completely (TPR 0.000). No one-sided behavior has been OBSERVED.

The gap in our knowledge: we have never measured the asymmetric operating
points — exact-FPR=0 (any TPR > 0 = gate candidate) or TPR=1.0 with the
smallest achievable FPR (pass-gate candidate). Both are one-line computations
on cached scores when this gets scheduled. Also unmeasured: whether the 43%
coverage (short docs escape) breaks any gate claim — a gate that can't fire
on short docs needs the panel to own short docs entirely.

Verdict: promising enough to keep on the board; unproven until extreme-point
measurements happen. When attacked, lead with min/max statistics and exact
zero-FPR evaluation on dev, then holdout confirmation (once, clean).

## Addendum (G5 jumble attack + CK2 decomposition, 2026-08-15)

- Content blindness verified: char-within-token jumble leaves the symbol
  stream byte-identical (1000/1000). The map is a pure function of token
  lengths.
- CK2 on these streams is mostly a MULTISET comparison: pct_ordered ~0.95
  under both jumble types (coarse 4-bit symbols => skewed distribution =>
  cheap nearest-neighbor matching blunts the order channel). Distances ride
  the pct_similar_chars channel (0.30 token-jumble, 0.18 char-global).
  Order-sensitive variants would need finer quantization.
- w32 stepped stat on dev docs: finite for 4% of docs, ALL human => at w32
  the stat is a doc-length proxy, not rhythm. Use w8 or stream-relative windows.
