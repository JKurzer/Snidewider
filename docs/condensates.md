# Statistical condensates — MS vectors, repetitiveness measures, SBWT

Distilled 2026-08-16 from papers/lit (see papers/MANIFEST.md):
Lipták/Masillo/Puglisi "Matching Statistics — a survey" (TCS 2026);
Bannai-circle/Puglisi "Sensitivity of Repetitiveness Measures to String
Reversal" (CPM 2026); Alanko/Biagi/Mackenzie/Puglisi "Batched k-mer Lookup
on the Spectral BWT" (ALENEX 2025). Donk's hunch made literal: succinct
structures ARE statistical condensates.

## 1. Matching statistics as a detector feature family

**Definition** (Chang & Lawler 1990): for strings S (test doc) and R
(reference corpus), MS[i] = (p_i, l_i) where l_i = length of the longest
prefix of S[i..] that occurs ANYWHERE in R. One pass over S, one integer
per position: "how much of what I'm reading exists in the reference."

**Key property** (Folklore Lemma 1): l_{i+1} >= l_i - 1. The vector is
nearly Lipschitz — smooth, compressible, and fast to compute (suffix
automaton/tree of R, then stream S: O(|R|+|S|)).

**Detection precedent inside the survey**: Ulitsky et al. use mean(l_i)/|X|
as a pairwise sequence DISTANCE for phylogenetics. So mean-MS-vs-reference
is a published similarity measure, not our invention.

**Feature design (ms_ family)** — doc vs pooled A-human reference:
- ms_mean, ms_median, ms_p90 of l_i (length of human-matching stretches)
- ms_frac0: fraction of positions with l_i = 0 (never-seen territory)
- ms_frac_ge16 / ge32: fraction covered by long matches
- Open question the fleet must answer: which direction separates? AI text
  reproduces high-frequency human substrings (longer l_i, less zero territory) OR
  drifts into model-specific phrase space (shorter). Either way it separates.
- Variant: MS vs the AI reference, and the contrast (same shape as
  cov*_contrast, which WORKS — 0.84 solo AUROC).

**Build cost**: suffix automaton over ~2MB human mega-string (one-time,
minutes in native; SAM gives longest-match lengths directly, positions not
needed for features).

## 2. Repetitiveness measures (the δ family)

The reversal paper catalogs measures and their symmetry robustness:
- **δ = max_k (distinct substrings of length k)/k** — REVERSAL-INVARIANT
  (reversing preserves the distinct-substring count), cheap, and we already
  compute q-gram profiles: distinct-counts at k=2..5 approximate the
  maximum's rising edge. THE feature: AI text is more repetitive → lower δ.
- r (BWT runs) — NOT reversal-stable; skip unless we want a second opinion.
- γ (string attractor size) — beautiful, expensive; skip for now.
- e (CDAWG edge count) — suffix-automaton-adjacent; free rider IF we build
  the SAM for MS vectors anyway. Same-structure bonus stat.

Design: col_delta2..5 (per-q distinct/k) + col_delta_max of the sampled
curve. Single pass, no comparisons — fits the distillation doctrine.

## 3. SBWT — the scale path, not a feature

Spectral-BWT batched k-mer lookup (Alanko et al.): compressed index over a
giant reference answering millions of membership queries. Our coverage
features currently hold the A-reference in Python Counters; that tops out
around a few MB of reference. If the reference grows to the full human
corpus, SBWT (or an r-index) is the structure. No feature semantics change
— same statistics, bigger R. Keep on the infrastructure shelf until the
reference outgrows RAM-resident Counters.

## Action mapping (docs/TODO.md #2)

1. ms_ family: suffix automaton over human mega-ref + walk each doc.
2. δ family: extend existing q-gram profile pass — nearly free.
3. e (CDAWG edges) if the SAM gets built for (1).
