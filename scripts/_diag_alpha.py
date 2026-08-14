"""Throwaway diagnostic: why does Hanada time grow with k?
Measures time vs k at fixed m, plus an empirical alpha estimate
(fraction of steps whose change point lands in the new scope).
"""

import random
import timeit

from ai_text_detection import qgram

rng = random.Random(7)
T = bytes(rng.randrange(97, 117) for _ in range(100_000))
M, Q = 500, 5
P = T[12345 : 12345 + M]

print("  k   hanada ms  ukkonen ms")
for k in (5, 50, 100, 250, 495):
    th = timeit.timeit(lambda: qgram.search(T, P, Q, k), number=1)
    tu = timeit.timeit(lambda: qgram.search_ukkonen(T, P, Q, k), number=1)
    print(f"{k:>4} {th * 1e3:>11.1f} {tu * 1e3:>12.1f}")

# empirical alpha: fraction of i where c_i in scope(i+1), k=495
K = 495
pat: dict[bytes, int] = {}
for i in range(len(P) - Q + 1):
    g = P[i : i + Q]
    pat[g] = pat.get(g, 0) + 1
occ: dict[bytes, list[int]] = {}
for i in range(len(T) - Q + 1):
    g = T[i : i + Q]
    occ.setdefault(g, []).append(i)

seen: dict[bytes, int] = {}
inside = 0
total = len(T) - Q
for cur in range(total):
    s = T[cur : cur + Q]
    ms = pat.get(s, 0)
    L = occ[s]
    idx = seen.get(s, 0) + ms
    seen[s] = seen.get(s, 0) + 1
    c = (L[idx] + Q - 1) if idx < len(L) else 1 << 60
    i = cur + 1
    b, e = i + M - 1 - K, min(i + M - 1 + K, len(T) - 1)
    if b <= c <= e:
        inside += 1
print(f"empirical alpha at m={M}, k={K}: {inside}/{total} = {inside / total:.5f}")
