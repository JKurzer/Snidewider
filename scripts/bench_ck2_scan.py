"""CK2 as a scanning primitive vs Hanada search — same text, same patterns.

The closest CK2 application to Hanada's substring search: a naive sliding
window. For every start i, score the |p|-length window against the pattern
with CK2 (0.0 = identical) and keep the minimum. Same inputs, same machine.

Caveats (not apples-to-apples): CK2 answers a different question (pairwise
order-sensitive score over a fixed window; no threshold enumeration, no
scope logic, no longest-on-ties). This bench quantifies the raw speed
trade of "algorithm" (Hanada) vs "fast primitive, no algorithm" (CK2).

Usage: .venv\\Scripts\\python scripts/bench_ck2_scan.py
"""

import random
import timeit

from ai_text_detection import ck2, qgram

rng = random.Random(7)
N, SIGMA, Q = 100_000, 20, 5
T = bytes(rng.randrange(97, 97 + SIGMA) for _ in range(N))
TV = memoryview(T)


def ck2_scan(p: bytes) -> float:
    m = len(p)
    best = 1.0
    for i in range(len(TV) - m + 1):
        score = ck2.similarity(TV[i : i + m], p)
        if score < best:
            best = score
    return best


def main() -> None:
    print(f"|t|={N}, sliding |p|-window CK2 scan vs Hanada q={Q} search (k=|p|-q)")
    print(f"{'|p|':>5} {'hanada ms':>10} {'ck2 scan ms':>12} {'ck2/hanada':>11} {'ck2 min':>8}")
    for m in (10, 50, 100, 200, 500):
        start = rng.randrange(N - m - 1)
        p = T[start : start + m]
        t_han = timeit.timeit(lambda: qgram.search(T, p, Q, m - Q), number=1)
        t_ck2 = timeit.timeit(lambda: ck2_scan(p), number=1)
        best = ck2_scan(p)  # separate untimed pass for the sanity value
        print(f"{m:>5} {t_han * 1e3:>10.1f} {t_ck2 * 1e3:>12.1f} {t_ck2 / t_han:>10.1f}x {best:>8.3f}")


if __name__ == "__main__":
    main()
