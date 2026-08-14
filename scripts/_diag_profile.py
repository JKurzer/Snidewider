"""Throwaway profiler driver: path-counter breakdown vs k (PERF-RULES #2)."""

import random

from ai_text_detection import _qgram_native

rng = random.Random(7)
T = bytes(rng.randrange(97, 117) for _ in range(100_000))
M, Q = 500, 5
P = T[12345 : 12345 + M]

COLS = (
    "hits", "steps", "full_updates", "baseline_steps", "corner_fixes",
    "update_iters", "argmin_iters", "edge_computes", "init_ms", "loop_ms",
)
rows = {}
for k in (5, 50, 100, 250, 495):
    rows[k] = _qgram_native.search_debug(T, P, Q, k)

print(f"{'k':>5} " + " ".join(f"{c:>14}" for c in COLS))
for k, r in rows.items():
    print(f"{k:>5} " + " ".join(f"{r[c]:>14.1f}" if isinstance(r[c], float) else f"{r[c]:>14}" for c in COLS))
