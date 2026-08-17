"""Wire the token-bigram peak-reuse pack into both caches (221 -> 226).

Evidence (fleet_token_bigrams): reuse features have real TPR@1e-2 tails
(0.13-0.17); the top-64 catalog rates have zero tails and dilute. Only the
reuse pack is wired. Idempotent. Usage:
.venv\\Scripts\\python scripts\\wire_reuse_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.token_bigrams import REUSE_FEATURE_NAMES, token_reuse_features

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
PREFIX = "reuse_"


def matrix(texts) -> np.ndarray:
    return np.array([[token_reuse_features(str(t))[k] for k in REUSE_FEATURE_NAMES]
                     for t in texts], dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if not n.startswith(PREFIX)]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + [PREFIX + n for n in REUSE_FEATURE_NAMES])
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
