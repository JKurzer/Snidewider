"""Wire the raw char-bigram rate vector into both caches (157 -> 221).

Donk's call: all three char-distribution families wired (charstat, collapse,
bigrams), tie or no tie. Idempotent: re-running replaces the bg_ columns.
Usage: .venv\\Scripts\\python scripts\\wire_bigram_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.bigrams import BIGRAM_FEATURE_NAMES, bigram_rates
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"


def bg_matrix(texts) -> np.ndarray:
    rows = []
    for t in texts:
        r = bigram_rates(str(t))
        rows.append([r[k] for k in BIGRAM_FEATURE_NAMES])
    return np.array(rows, dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if not n.startswith("bg_")]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + list(BIGRAM_FEATURE_NAMES))
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features", flush=True)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    dev_new = {f"X_{b}": bg_matrix(buckets[b].generation) for b in "ABC"}
    for b in "ABC":
        print(f"dev {b} done", flush=True)
    rewire(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_new = {
        "X_hu": bg_matrix(hold[hold.model == "human"].generation),
        "X_ai": bg_matrix(hold[hold.model != "human"].sample(n=20_000, random_state=97)
                           .generation),
    }
    rewire(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
