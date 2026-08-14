"""Quick-and-dirty CK2-vs-rapidfuzz timing sanity check.

Not a rigorous benchmark (no warmup discipline, shared RNG state across
timers): it exists to sanity-check the orders of magnitude behind the
'3x faster short / 300x+ faster long' claim from the CKS Readme.
Usage: .venv\\Scripts\\python scripts/bench_ck2.py
"""

import random
import timeit

from rapidfuzz.distance import Levenshtein

from ai_text_detection import ck2

rng = random.Random(42)


def rand_pair(n: int) -> tuple[bytes, bytes]:
    a = bytes(rng.randrange(256) for _ in range(n))
    # sibling string: mutate ~5% of positions so scores are interesting
    b = bytearray(a)
    for _ in range(max(1, n // 20)):
        b[rng.randrange(n)] = rng.randrange(256)
    return a, bytes(b)


def bench(label: str, fn, pairs, number: int) -> None:
    t = timeit.timeit(lambda: [fn(a, b) for a, b in pairs], number=number)
    per_call = t / (number * len(pairs)) * 1e6
    print(f"  {label:<28} {per_call:>12.2f} us/pair")


def main() -> None:
    for size, npairs, number in ((50, 200, 20), (2_000, 5, 20), (20_000, 2, 3)):
        pairs = [rand_pair(size) for _ in range(npairs)]
        print(f"string length {size} B  ({npairs} pairs x{number})")
        bench("ck2 native", ck2.similarity, pairs, number)
        bench("rapidfuzz Levenshtein", Levenshtein.distance, pairs, number)


if __name__ == "__main__":
    main()
