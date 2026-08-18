"""FLEET W — bug-check the ApEn/SampEn implementation.

Oracle: brute-force O(N^2) block-matching (no KD-tree) on short docs.
Sanity: constant -> 0, periodic -> low, random -> high, prose mid.
Also: length-dependence curve (same text truncated).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fleet_apen import _block, apen_features  # noqa: E402

from ai_text_detection.evaldata import split_buckets  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def oracle_apen(series: np.ndarray, m: int, r: float) -> tuple[float, float]:
    """Textbook brute force: all-pairs Chebyshev on m-blocks."""
    phis = []
    pair_counts = []
    for mm in (m, m + 1):
        X = _block(series, mm)
        d = np.abs(X[:, None, :] - X[None, :, :]).max(axis=2)
        matches = (d <= r)
        c = matches.sum(axis=1) / len(X)          # includes self-match
        phis.append(np.log(c).mean())
        pair_counts.append((matches.sum() - len(X)) / 2)  # i != j pairs
    apen = float(phis[0] - phis[1])
    b_p, a_p = pair_counts
    sampen = float(-np.log(a_p / b_p)) if a_p > 0 and b_p > 0 else np.nan
    return apen, sampen


def main() -> None:
    print("== 1. canonical sanity ==", flush=True)
    const = np.full(500, 97.0)
    periodic = np.tile(np.array([97.0, 98.0, 99.0, 100.0]), 125)
    rng = np.random.default_rng(7)
    random_s = rng.uniform(32, 126, 500)
    a_const, _ = oracle_apen(const, 2, 0.2 * const.std() or 1e-9)
    check("constant ApEn ~ 0", a_const < 0.01, f"({a_const:.4f})")
    a_per, _ = oracle_apen(periodic, 2, 0.2 * periodic.std())
    a_rand, _ = oracle_apen(random_s, 2, 0.2 * random_s.std())
    check("periodic < random", a_per < a_rand,
          f"(periodic {a_per:.3f} < random {a_rand:.3f})")

    print("== 2. KD-tree vs brute-force oracle, 12 real docs ==", flush=True)
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    c = split_buckets(df)["C"]
    gens = [str(t) for t in c.generation[:12]]
    worst = 0.0
    for i, text in enumerate(gens):
        b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)[:600]
        r = 0.2 * float(b.std())
        o_apen, o_sampen = oracle_apen(b, 2, r)
        kd = apen_features(text[:600])
        d_apen = abs(kd["apen_char"] - o_apen)
        d_sampen = abs(kd["sampen_char"] - o_sampen)
        worst = max(worst, d_apen, d_sampen if np.isfinite(d_sampen) else 0.0)
    check("KD == oracle (12 docs, 1e-9)", worst < 1e-9, f"(worst |d| = {worst:.2e})")

    print("== 3. length dependence (same doc truncated) ==", flush=True)
    text = gens[0]
    for n in (300, 600, 1200, 2400):
        v = apen_features(text[:n])["apen_char"]
        print(f"  N={n:>5}: apen {v:.4f}", flush=True)

    print(f"\n{'ALL PASS' if not FAIL else f'FAILURES: {FAIL}'}")


if __name__ == "__main__":
    main()
