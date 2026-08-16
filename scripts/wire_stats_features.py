"""Wire classical stats + coverage features into both caches (89 -> 120).

Appends 22 stat_* + 9 cov* columns. Idempotent: re-running replaces the
stat_/cov columns rather than duplicating. Layout stays: base 0:81,
shape 81:89, stats 89:111, coverage 111:120.

Coverage references are pooled gram counters built CROSS-BUCKET (fleet G
probe: A rows scored vs A-built refs are in-fold inflated even with
doc/source LOU — the reference bucket itself is the skew; hgb_b-on-A read
0.999 vs hgb_a-on-C 0.910). So: A rows score against refs from B+C,
B/C/holdout rows against refs from A. Every row faces a foreign reference;
no leave-one-out needed at all.

Usage: .venv\\Scripts\\python scripts\\wire_stats_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.coverage import (
    COVERAGE_FEATURE_NAMES,
    QS,
    build_reference,
    coverage_features,
    source_exclusion,
)
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.stats_features import STAT_FEATURE_NAMES, stat_features

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
NEW_NAMES = [f"stat_{n}" for n in STAT_FEATURE_NAMES] + list(COVERAGE_FEATURE_NAMES)


def stats_matrix(texts) -> np.ndarray:
    rows = [[stat_features(str(t))[k] for k in STAT_FEATURE_NAMES] for t in texts]
    return np.array(rows, dtype=float)


def coverage_matrix(sub: pd.DataFrame, ref_hu, ref_ai, loo: bool) -> np.ndarray:
    rows = []
    if not loo:
        for t in sub.generation:
            f = coverage_features(str(t), ref_hu, ref_ai)
            rows.append([f[k] for k in COVERAGE_FEATURE_NAMES])
        return np.array(rows, dtype=float)
    # source-level leave-one-out: exclusion = all grams of the source's rows
    texts = [str(t) for t in sub.generation]
    sources = list(sub.source_id)
    excl_by_source: dict[str, dict[int, "Counter"]] = {}
    for src in set(sources):
        mate_texts = [texts[i] for i, s in enumerate(sources) if s == src]
        excl_by_source[src] = {q: source_exclusion(mate_texts, q) for q in QS}
    for text, src in zip(texts, sources):
        f = coverage_features(text, ref_hu, ref_ai, exclude=excl_by_source[src])
        rows.append([f[k] for k in COVERAGE_FEATURE_NAMES])
    return np.array(rows, dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"]
             if not (n.startswith("stat_") or n.startswith("cov"))]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + NEW_NAMES)
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features")


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    a = buckets["A"]
    # cross-bucket references: nothing scores against its own fold's grams
    ref_a = ({q: build_reference(a[a.model == "human"].generation, q) for q in QS},
             {q: build_reference(a[a.model != "human"].generation, q) for q in QS})
    bc = pd.concat([buckets["B"], buckets["C"]])
    ref_bc = ({q: build_reference(bc[bc.model == "human"].generation, q) for q in QS},
              {q: build_reference(bc[bc.model != "human"].generation, q) for q in QS})
    print("refs built (cross-bucket)", flush=True)

    dev_new = {}
    for b in "ABC":
        rhu, rai = (ref_bc if b == "A" else ref_a)
        dev_new[f"X_{b}"] = np.column_stack([
            stats_matrix(buckets[b].generation),
            coverage_matrix(buckets[b], rhu, rai, loo=False),
        ])
        print(f"dev {b}: {dev_new[f'X_{b}'].shape}", flush=True)
    rewire(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_hu = hold[hold.model == "human"]
    hold_ai = hold[hold.model != "human"].sample(n=20_000, random_state=97)
    hold_new = {}
    for key, sub in (("X_hu", hold_hu), ("X_ai", hold_ai)):
        hold_new[key] = np.column_stack([
            stats_matrix(sub.generation),
            coverage_matrix(sub, ref_a[0], ref_a[1], loo=False),
        ])
        print(f"holdout {key}: {hold_new[key].shape}", flush=True)
    rewire(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
