"""Wire the shape family (skeleton + dct_run CK2 stats) into the feature panel.

Appends the 8 shape features to both caches (dev full_features.npz,
holdout_features.npz) — computed fresh per doc, joined to the cached columns.
Idempotent: re-running replaces the shape columns rather than duplicating.
Usage: .venv\\Scripts\\python scripts/wire_shape_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.shape import SHAPE_FEATURE_NAMES, shape_features

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
SHAPE_NAMES = [f"shape_{n}" for n in SHAPE_FEATURE_NAMES]


def shape_matrix(texts) -> np.ndarray:
    rows = [[shape_features(str(t))[k] for k in SHAPE_FEATURE_NAMES] for t in texts]
    return np.array(rows, dtype=float)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")

    dev_cache = np.load(DEV_NPZ)
    names = [n for n in dev_cache["feature_names"] if not n.startswith("shape_")]
    store = {k: dev_cache[k] for k in dev_cache if k != "feature_names"}
    for b in ("A", "B", "C"):
        store[f"X_{b}"] = store[f"X_{b}"][:, : len(names)]
    buckets = split_buckets(df)
    for b in ("A", "B", "C"):
        sm = shape_matrix(buckets[b].generation)
        assert len(sm) == len(store[f"X_{b}"])
        store[f"X_{b}"] = np.column_stack([store[f"X_{b}"], sm])
        print(f"dev {b}: {store[f'X_{b}'].shape}")
    store["feature_names"] = np.array(names + SHAPE_NAMES)
    np.savez(DEV_NPZ, **store)

    hold_cache = np.load(HOLD_NPZ)
    store = {"X_hu": hold_cache["X_hu"][:, : len(names)], "X_ai": hold_cache["X_ai"][:, : len(names)]}
    hold = df[df.fold == "holdout"]
    hold_hu = hold[hold.model == "human"]
    hold_ai = hold[hold.model != "human"].sample(n=20_000, random_state=97)
    for key, sub in (("X_hu", hold_hu), ("X_ai", hold_ai)):
        sm = shape_matrix(sub.generation)
        assert len(sm) == len(store[key])
        store[key] = np.column_stack([store[key], sm])
        print(f"holdout {key}: {store[key].shape}")
    store["feature_names"] = np.array(names + SHAPE_NAMES)
    np.savez(HOLD_NPZ, **store)
    print(f"both caches now at {len(names + SHAPE_NAMES)} features")


if __name__ == "__main__":
    main()
