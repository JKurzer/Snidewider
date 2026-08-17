"""FLEET E2 — PRODUCTION featurize (pipeline.py + detector bundle) vs cache.

Verifies the artifact we'd ship: pipeline.featurize on bucket C rows must
match the cached columns bit-exactly (C uses production reference semantics:
A-built coverage refs, no bank membership). Bucket A's cache columns differ
by design (cross-bucket coverage refs for honest training) and are NOT the
production path.

Also: exemplar-bank LOO assertion on A row 0, purity, NaN-rate audit.
Usage: .venv\\Scripts\\python scripts\\exp\\fleet_e2_cache_truth.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import pipeline
from ai_text_detection.evaldata import split_buckets

SAMPLE = 60
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    cache = np.load("data/derived/full_features.npz")
    names = list(cache["feature_names"])
    artifacts = pipeline.load_artifacts()

    print("== exemplar LOO: bank members must NOT self-match ==", flush=True)
    Xa = cache["X_A"]
    hu_min_col = names.index("ex_hu_min")
    check("row 0 (bank_hu member) ex_hu_min > 0", Xa[0, hu_min_col] > 0,
          f"(got {Xa[0, hu_min_col]:.4f})")

    print("== PRODUCTION pipeline.featurize (csa_mode=full) vs cache, 60 C rows ==",
          flush=True)
    cb = buckets["C"]
    Xc = cache["X_C"]
    idx = np.linspace(0, len(cb) - 1, SAMPLE).astype(int)
    gens = list(cb.generation)
    mismatch = 0
    for i in idx:
        fresh = pipeline.featurize(str(gens[i]), artifacts, csa_mode="full")
        cached = Xc[i]
        if not np.allclose(np.nan_to_num(fresh), np.nan_to_num(cached), atol=1e-9):
            d = np.abs(np.nan_to_num(fresh) - np.nan_to_num(cached))
            mismatch += 1
            print(f"    row {i}: max |diff| {d.max():.3e} at {names[int(d.argmax())]}",
                  flush=True)
    check(f"pipeline==cache (60 C rows x {Xc.shape[1]} feats)", mismatch == 0,
          f"({mismatch} mismatched rows)")

    print("== purity: same doc twice (production path) ==", flush=True)
    twice = pipeline.featurize(str(gens[idx[0]]), artifacts, csa_mode="full")
    check("pure function", np.array_equal(
        np.nan_to_num(twice),
        np.nan_to_num(pipeline.featurize(str(gens[idx[0]]), artifacts, csa_mode="full"))))

    print("== impute mode: csa_* columns come from A-means ==", flush=True)
    names0 = list(cache["feature_names"])
    csa_idx = [names0.index(f"csa_{k}") for k in ("n", "wt_rate", "sada_rate")]
    cheap = pipeline.featurize(str(gens[idx[1]]), artifacts, csa_mode="impute")
    want = artifacts["impute_means"][csa_idx]
    check("impute mode fills csa cols from A-means",
          bool(np.allclose(cheap[csa_idx], want, atol=1e-12)))

    print("== NaN-rate audit per bucket ==", flush=True)
    for b in "ABC":
        X = cache[f"X_{b}"]
        rates = np.isnan(X).mean(axis=0)
        worst = rates.argsort()[-5:][::-1]
        top = ", ".join(f"{names[i]} {rates[i]:.2f}" for i in worst if rates[i] > 0)
        print(f"  {b}: {int((rates > 0).sum())} cols with NaN; worst: {top}", flush=True)

    print(f"\n{'ALL PASS' if not FAIL else f'FAILURES: {FAIL}'}")


if __name__ == "__main__":
    main()
