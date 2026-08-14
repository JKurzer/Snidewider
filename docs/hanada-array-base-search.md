# VERIFIED (see docs/hanada-verification.md; fix in docs/hanada-deque-fix.md)

> All claims independently verified against the paper 2026-08-14 (6/6
> CONFIRMED). The corner is real, the mechanism holds, and the fix (monotone
> deque = the paper's own DEL-FIRST idea) lands in docs/hanada-deque-fix.md.

# Hanada Array+Base-Search — profiling findings (2026-08-14)

## Context

Implementation of Array+Base-Search (Hanada, Kudo, Nakamura 2014, Fig. 5) in
`src/ai_text_detection/native/qgram_module.cpp`, oracle-verified
(`tests/test_hanada.py`, 28/28 green at time of writing). Paper:
`papers/hanada2014-qgram-search.pdf`.

## Symptom

Bench (`scripts/bench_hanada.py`; paper settings |t|=100,000, |Σ|=20, q=5,
k=|p|−q, patterns drawn from t): Hanada only 1.2–1.7x faster than the Ukkonen
baseline, and time grows ~linearly in k. Expected average-case O(|t|+|p|),
i.e. ~flat in |p|.

## Measurements (PERF-RULES: instrumented, not guessed)

`search_debug` path counters at m=500, q=5:

| k   | full_updates | corner_fixes | argmin_iters | loop_ms |
|-----|--------------|--------------|--------------|---------|
| 5   | 0            | 98,980       | 890,785      | 23.6    |
| 100 | 0            | 98,980       | 19,677,320   | 66.5    |
| 495 | 0            | 98,980       | 97,402,655   | 246.2   |

- α = 0 measured independently (`scripts/_diag_alpha.py`): the full-update
  path never fires; the baseline skip works as designed (update_iters = 0).
- The j\*-at-erased-edge corner fix fires on 99.4% of steps; each fires an
  O(k) argmin. argmin_iters tracks loop_ms exactly.

## Mechanism

With random text and rare pattern-gram matches, d(i)(j) is ~monotonically
increasing in j (a longer window mostly adds excess grams), so j\* sits at
the left scope edge ~always; the scope slides right every step and evicts
it. The no-local-match regime is ~99% of positions in real search workloads.

## Claim about the paper

Fig. 5's core premise — "if c_i is outside scope(i+1), then j\* is unchanged
or becomes e_{i+1}" (paper §4.1) — is false in this regime; the paper never
recomputes j\* on the baseline path. Our corner fix is required for
correctness (oracle-verified), and its O(k) cost explains both our growth
and, plausibly, the paper's own unexplained Array+Base growth in Fig. 14(a)
(which the authors attribute to the q-condition instead).

## Proposed fix

Sliding-window min with frozen relative order on the baseline path:
monotone deque, amortized O(1)/step, rebuilt O(k) only on full updates
(α·k amortized). This is the candidate-list idea from the paper's own
List+Base-Search (§4.3) applied to the Array variant.
