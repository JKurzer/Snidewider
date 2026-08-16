# G5 — dct_run_map jumble-blindness: assumption-check attack

**Claim under attack:** the `dct_run_map` CK2 family measures *rhythm*
(token-length texture), not *content*. If true: jumbling characters inside
tokens must be invisible; jumbling token order should scramble the raw
stream but not what the downstream stats capture; destroying token lengths
must register.

**Data:** dev fold only (`raid_splits.parquet`, `fold=='dev'`), 500 human +
500 AI, `random_state=61`. Per-doc seeds (`g5:61:<id>`) — order-independent,
pure. No holdout touched; nothing tuned (RULES #4).

## Measured caveat: the spec'd w32 stat barely exists on dev

`dct_run_map` emits 1 symbol per 8 tokens; median dev doc → 29-symbol
stream. Stepped CK2 at window 32 needs ≥64 symbols:

- **w32 finite for 40/1000 docs (4.0%) — and all 40 are human.** On dev,
  `dct_run_step_stdev` at w32 is effectively a long-doc/human proxy, not a
  rhythm feature.
- w8 supplement (finite 880/1000 = 88%) used for stats-movement claims.
- CK2 stream-vs-stream distances are length-agnostic → full 1000-doc coverage.

## Results

| # | Probe | Headline number | Verdict |
|---|-------|-----------------|---------|
| 2 | TOKEN-JUMBLE: per-doc CK2(orig, jumbled) | **0.697** mean / 0.692 median / max 1.0 | raw stream NOT blind to order (expected: same symbol multiset ⇒ CK2 isolates order; permuted order scores high) |
| 2 | TOKEN-JUMBLE: stats movement (w8) | mean \|Δstdev\| **0.0377** vs baseline 0.0818 (±46% jitter), signed Δ **−0.0010** (no direction) | stat jitters under permutation but drifts nowhere |
| 2 | class separation before → after | w8 AUROC **0.512 → 0.511** (n=880) | the (null) authorship signal is fully preserved under jumbling |
| 3 | CHAR-JUMBLE-WITHIN-TOKENS: byte-identical streams | **1000/1000 = 1.0000** [Wilson 0.9962, 1.0] | total content blindness — the map is a pure function of token lengths |
| 4 | CHAR-JUMBLE-GLOBAL: CK2(orig, global) | **0.814** mean; w8 \|Δstdev\| 0.0368 | map DOES respond when lengths are destroyed — sanity holds |
| 5 | stdev AUROC: original-vs-token-jumbled vs human-vs-AI | w8: **0.494** (n=1760) vs **0.512** (n=880); w32: 0.507 (n=80) vs nan (no AI coverage) | NOT more sensitive to jumbling than to authorship — both are coin flips (SE ≈ 0.014–0.02) |

## Verdict

**Blindness assumption HOLDS.** Character content is provably invisible
(1000/1000 byte-identical). Token order strongly perturbs the raw stream
(CK2 0.697 — that is CK2 doing its job as an order metric over an unchanged
symbol multiset) but the summary stats move only as zero-mean jitter, and
what little class separation exists is untouched. The feature family reads
rhythm, not content — and on dev docs it currently reads *no* authorship
signal either (w8 AUROC ≈ 0.51).

**Side finding for the shape family:** the spec'd w32 stepped config is
unmeasurable on 96% of dev docs, and its survivors are exclusively human —
any w32 dct_run "separation" on dev is a doc-length artifact. Gate candidates
from this family need a window sized to the stream (≤8) or length-matched
evaluation.

Repro: `scripts/exp_g5.py` (run from the main checkout cwd with
`PYTHONPATH=<worktree>\src`; natives pre-staged).
