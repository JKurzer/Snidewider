"""Wire oct_hits (Donk's cumulative octgram recurrence) into both caches.

One column only, per order. Idempotent: strips oct_ columns, appends fresh.
Usage: .venv\\Scripts\\python scripts\\wire_oct_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.token_bigrams import oct_hits_features

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"


def col(texts) -> np.ndarray:
    return np.array([oct_hits_features(str(t))["oct_hits"] for t in texts], dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if not n.startswith("oct_")]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + ["oct_hits"])
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features", flush=True)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    dev_new = {f"X_{b}": col(buckets[b].generation) for b in "ABC"}
    for b in "ABC":
        print(f"dev {b} done", flush=True)
    rewire(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_new = {
        "X_hu": col(hold[hold.model == "human"].generation),
        "X_ai": col(hold[hold.model != "human"].sample(n=20_000, random_state=97)
                    .generation),
    }
    rewire(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
