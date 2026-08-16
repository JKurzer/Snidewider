"""FLEET E2 — fresh featurize vs cached columns, all families (dev A sample).

The 0.714-vs-0.7113 exam/ladder gap was never root-caused. If cached feature
columns are bit-identical to a fresh featurize of the same rows, the gap was
model-level (HGB binning on different-yet-equivalent data is deterministic,
so it would implicate sampling); if not, we have a cache bug. Also: feature
purity (same doc twice -> identical) and per-bucket NaN-rate audit.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_e2_cache_truth.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import qgram
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import ExemplarBank, exemplar_vector
from ai_text_detection.feature_sets import qgram12_vector, relative_vector
from ai_text_detection.shape import SHAPE_FEATURE_NAMES, shape_features

N_BANK = 150
SAMPLE = 60
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def featurize_all(text: str, bank_ai: ExemplarBank, bank_hu: ExemplarBank) -> list[float]:
    tail = dct_tail_features(text)
    shape = shape_features(text)
    return (
        relative_vector(text)
        + qgram12_vector(text)
        + exemplar_vector(qgram.profile(text.encode("utf-8"), 3), bank_ai, bank_hu)
        + [tail[k] for k in sorted(tail)]
        + [shape[k] for k in SHAPE_FEATURE_NAMES]
    )


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    a = buckets["A"]
    bank_ai = ExemplarBank.from_texts([str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts([str(t) for t in a[a.model == "human"].generation[:N_BANK]])

    cache = np.load("data/derived/full_features.npz")
    Xa = cache["X_A"]

    print("== fresh vs cache, 60 A rows (sampled across the bucket) ==")
    idx = np.linspace(0, len(a) - 1, SAMPLE).astype(int)
    gens = list(a.generation)
    mismatch = 0
    for i in idx:
        fresh = np.array(featurize_all(str(gens[i]), bank_ai, bank_hu), dtype=float)
        cached = Xa[i]
        if not np.allclose(np.nan_to_num(fresh), np.nan_to_num(cached), atol=1e-9):
            d = np.abs(np.nan_to_num(fresh) - np.nan_to_num(cached))
            mismatch += 1
            print(f"    row {i}: max |diff| {d.max():.3e} at col {int(d.argmax())}", flush=True)
    check("fresh==cache (60 rows x 89 feats)", mismatch == 0, f"({mismatch} mismatched rows)")

    print("== purity: same doc twice ==")
    twice = featurize_all(str(gens[idx[0]]), bank_ai, bank_hu)
    check("pure function", np.array_equal(np.nan_to_num(twice), np.nan_to_num(
        featurize_all(str(gens[idx[0]]), bank_ai, bank_hu))))

    print("== NaN-rate audit per bucket ==")
    names = list(cache["feature_names"])
    for b in "ABC":
        X = cache[f"X_{b}"]
        rates = np.isnan(X).mean(axis=0)
        worst = rates.argsort()[-5:][::-1]
        top = ", ".join(f"{names[i]} {rates[i]:.2f}" for i in worst if rates[i] > 0)
        print(f"  {b}: {int((rates > 0).sum())} cols with NaN; worst: {top}", flush=True)


if __name__ == "__main__":
    main()
