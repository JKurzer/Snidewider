# Review — Equi, Khan, Mäkinen (2026), Quantum Pattern Matching in
# Generalized Degenerate Strings (arXiv:2603.16297)

Reviewed 2026-08-16, explicitly through the entropy lens Donk ordered.
Verdict up front: SHELF was the wrong call. This is a paper about
**computing over ensembles**, and an ensemble is one logarithm away from an
entropy.

## What the paper is

- **Object**: a *generalized degenerate (GD) string* T = T[1]..T[n] where
  each segment T[i] is a SET of alternative strings. A GD string spells an
  ensemble of ordinary strings (choose one alternative per segment).
  N = total characters across all segments.
- **Problem** (SMGD): find pattern P in any string spelled by T.
- **Result**: first quantum algorithm for SMGD, Õ(sqrt(m·n·N)) time via
  nested Grover searches (outer over shifts, inner over mismatch checks),
  using a *multi-threaded abstraction*: each quantum thread owns one shift,
  so the algorithm reads like a classical parallel one. They note lower
  bounds block quantum advantage on plain strings — the GD structure is
  what creates the room. That is WHY they generalized: the interesting
  frontier is structured/uncertain strings, not flat ones.

## The entropy reading (the reason this matters to us)

A GD string is a **factorized statistical condensate**:
- |T[i]| alternatives at segment i → per-position entropy budget
  H_i = log2|T[i]| (uniform) — or the weighted version with probabilities.
- Ensemble size = prod |T[i]| → total entropy = sum of H_i.
- The paper's N is the *description length of the ensemble*; the runtime
  sqrt(m·n·N) is measured against that entropy volume.

Now replace "segment" with "token position in an LLM": the model's top-k
support at each position IS a degenerate segment. A generated document is
one path through a giant GD string. **Model collapse in GD terms = segment
sets shrinking = per-position entropy deflating.** The QPM formalism is the
discrete, set-valued version of what perplexity detectors measure
continuously. It is the right native frame for "text as a sample from a
concentrating distribution."

## What transfers to Snidewider

1. **Structure-size = entropy statistics.** Donk's remark, formalized: the
   size of a string's minimal index is a principled complexity measure:
   - suffix automaton state count (≈ CDAWG's e from the repetitiveness lit)
   - BWT run count r (directly a repetitiveness measure)
   - LZ77 phrase count z
   - δ = max_k distinct-k-mers/k (TODO #2, nearly free)
   These are the honest versions of our zlib_ratio proxy. A doc that
   compresses into a small structure is a doc from a concentrated
   distribution.
2. **The GD frame justifies per-position concentration features**: any stat
   that estimates the *support size* of the generating process (rare-word
   spectrum, coverage of the long tail — most of which we already ship in
   the collapse pack) is an estimate of the GD ensemble's entropy.
3. **Nothing quantum is needed.** The value here is the abstraction
   (ensembles as first-class strings) and the design pattern (thread-per-
   hypothesis decomposition), not the hardware.

## Why they bothered (trusting Doc Mak's judgment)

Quantum string matching is boxed in by conditional lower bounds on flat
strings; GD strings are where the structure gives quantum something to
exploit. Same lesson as our stack: **the money is in the structure of the
data, not in generic horsepower.** Also notable: they deliberately designed
for legibility ("accessible to non-quantum specialists") — a Mäkinen
trademark worth copying.

## Action

- papers/MANIFEST.md: QPM moved SHELF -> DISTILLED (this note).
- TODO #4a (new): structure-entropy pack — sam_states, bwt_runs, lz77_phrases,
  delta — as the principled upgrade of zlib_ratio. The SAM build for the
  ms_ family (condensates.md) yields sam_states as a free rider.
