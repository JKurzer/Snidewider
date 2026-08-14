# Hanada deque fix — resolution (2026-08-14)

Resolution of `docs/hanada-array-base-search.md` (NEEDS TO BE VERIFIED).
Independent verification: `docs/hanada-verification.md` (branch
verify-hanada-note) — all six claims CONFIRMED against the paper, with one
correction (paper's experiment uses k=|p|, not k=|p|−q; note amended) and one
sharpening: the paper handles the same corner in List+Base-Search's DEL-FIRST
but never backported it to Fig. 5.

## The fix

j\* tracking on the Hanada baseline path is now a monotone deque of
(j, offset) — sliding-window min, largest-j-on-ties, front pops when the
left edge leaves scope (structural DEL-FIRST analogue), back pops maintain
monotonicity. Rebuilt O(k) only on the full-update path (charged to α, which
measures 0 in the bench regime). The argmin corner fix is gone.

## Verification

- Oracle suite: 28/28 pass (`tests/test_hanada.py` — brute-force ground truth,
  incl. k=0 regression and both paper examples), run against the fix-worktree
  build with PYTHONPATH pinned (editable-install shadowing bit us once;
  imports verified by `__file__` print).
- Counters (`search_debug`): evictions 98,980 now O(1) each; back_pops ~1K
  total; argmin_iters 97,402,655 -> 1,981 at k=495 (init scans only).

## Before/after (scripts/bench_hanada.py, same machine/settings)

| k   | Hanada before (ms) | Hanada deque (ms) | Ukkonen (ms) |
|-----|--------------------|-------------------|--------------|
| 5   | 65.4               | 74.5              | 77.2         |
| 100 | 127.2              | 67.6              | 156.4        |
| 495 | 342.8              | 70.2              | 562.8        |

Hanada is now flat in k (~70-75 ms = the hash-map constant floor), 8x faster
than Ukkonen at k=495 and diverging. Known trade: at k<=10 the deque is
~10-30% slower than the old tiny argmin; crossover ~k=50. Remaining constant
is per-step unordered_map work (~700 ns/position) — the documented next
target if profiling ever makes it matter (PERF-RULES #2).
