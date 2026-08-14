"""q-gram native timing sanity check, incl. the cached-profile corpus pattern.
Usage: .venv\\Scripts\\python scripts/bench_qgram.py
"""

import random
import timeit

from ai_text_detection import qgram

rng = random.Random(42)


def oracle_distance(a: bytes, b: bytes, q: int) -> int:
    def prof(s: bytes) -> dict[bytes, int]:
        p: dict[bytes, int] = {}
        for i in range(len(s) - q + 1):
            g = s[i : i + q]
            p[g] = p.get(g, 0) + 1
        return p

    p1, p2 = prof(a), prof(b)
    return sum(abs(p1.get(g, 0) - p2.get(g, 0)) for g in p1.keys() | p2.keys())


def rand_pair(n: int, alphabet: int = 26) -> tuple[bytes, bytes]:
    a = bytes(rng.randrange(97, 97 + alphabet) for _ in range(n))
    b = bytearray(a)
    for _ in range(max(1, n // 20)):
        b[rng.randrange(n)] = rng.randrange(97, 97 + alphabet)
    return a, bytes(b)


def bench(label: str, fn, pairs, number: int) -> None:
    t = timeit.timeit(lambda: [fn(a, b) for a, b in pairs], number=number)
    print(f"  {label:<32} {t / (number * len(pairs)) * 1e6:>12.2f} us/pair")


def main() -> None:
    for size, npairs, number in ((50, 200, 20), (2_000, 5, 20), (20_000, 2, 3)):
        pairs = [rand_pair(size) for _ in range(npairs)]
        print(f"string length {size} B, q=3  ({npairs} pairs x{number})")
        bench("qgram native distance", lambda a, b: qgram.distance(a, b, 3), pairs, number)
        bench("naive python oracle", lambda a, b: oracle_distance(a, b, 3), pairs, max(1, number // 2))

    # corpus pattern: profile once, diff many
    docs = [rand_pair(2_000)[0] for _ in range(50)]
    suspect = docs[0]
    t = timeit.timeit(
        lambda: [qgram.distance_profiles(qgram.profile(suspect), qgram.profile(d)) for d in docs],
        number=5,
    )
    sp = qgram.profile(suspect)
    cached_profiles = [qgram.profile(d) for d in docs]
    t_cached = timeit.timeit(
        lambda: [qgram.distance_profiles(sp, qgram.profile(d)) for d in docs],
        number=5,
    )
    t_reuse = timeit.timeit(lambda: [qgram.distance_profiles(sp, dp) for dp in cached_profiles], number=5)
    print(f"50 x 2KB docs: one-shot {t/5:.3f}s | re-profiled {t_cached/5:.3f}s | cached profiles {t_reuse/5:.3f}s")


if __name__ == "__main__":
    main()
