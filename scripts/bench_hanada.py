"""Replicates Hanada et al. 2014 Fig. 14(a): computation time vs |p|.

Paper settings: |t|=100,000, |Sigma|=20, q=5, patterns are substrings of t.
Paper used k=|p|; we use k=|p|-q, the largest value inside the validated
regime (every in-scope substring >= q chars). Expect: Ukkonen grows ~linearly
in |p| (O(|t|k)); Hanada ~flat (average O(|t|+|p|)).

Usage: .venv\\Scripts\\python scripts/bench_hanada.py
"""

import random
import timeit

from ai_text_detection import qgram

rng = random.Random(7)

N, SIGMA, Q = 100_000, 20, 5
T = bytes(rng.randrange(97, 97 + SIGMA) for _ in range(N))


def main() -> None:
    print(f"|t|={N}, |Sigma|={SIGMA}, q={Q}, k=|p|-q; patterns = substrings of t")
    print(f"{'|p|':>5} {'k':>5} {'ukkonen ms':>12} {'hanada ms':>11} {'speedup':>8} {'hits':>7}")
    for m in (10, 50, 100, 200, 500):
        start = rng.randrange(N - m - 1)
        p = T[start : start + m]
        k = m - Q
        t_ukk = timeit.timeit(lambda: qgram.search_ukkonen(T, p, Q, k), number=1)
        t_han = timeit.timeit(lambda: qgram.search(T, p, Q, k), number=1)
        hits = len(qgram.search(T, p, Q, k))
        print(f"{m:>5} {k:>5} {t_ukk * 1e3:>12.1f} {t_han * 1e3:>11.1f} {t_ukk / t_han:>7.1f}x {hits:>7}")


if __name__ == "__main__":
    main()
