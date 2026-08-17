"""Wire the peak-reuse ceiling into both caches (226 -> 222, abs only).

Donk's trim: of the 5 reuse features, keep ONLY peak_reuse_abs (the rawest
statistic - absolute count of the top bigram, no rate-mixing). Idempotent:
strips all reuse_ columns and appends just reuse_peak_reuse_abs.
"""

import numpy as np
import pandas as pd

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.token_bigrams import REUSE_FEATURE_NAMES, token_reuse_features

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
PREFIX = "reuse_"


def matrix(texts) -> np.ndarray:
    return np.array([[token_reuse_features(str(t))["peak_reuse_abs"]]
                     for t in texts], dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if not n.startswith(PREFIX)]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + [PREFIX + "peak_reuse_abs"])
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features", flush=True)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    dev_new = {f"X_{b}": matrix(buckets[b].generation) for b in "ABC"}
    for b in "ABC":
        print(f"dev {b} done", flush=True)
    rewire(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_new = {
        "X_hu": matrix(hold[hold.model == "human"].generation),
        "X_ai": matrix(hold[hold.model != "human"].sample(n=20_000, random_state=97)
                       .generation),
    }
    rewire(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
