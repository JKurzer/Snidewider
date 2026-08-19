"""Wire the raw s256 self-similarity series into both caches (253 -> 509).

Fleet MOSAIC: panel+raw256 +0.020 @1e-2 / +0.055 @1e-3 over the 253 (largest
single addition of the project). Column names s256_000..s256_255 (series
order, content-hash-seeded per doc). Idempotent: strips s256_* columns,
appends fresh. Usage: .venv\\Scripts\\python scripts\\wire_raw256_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.burst import random_change_series
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
SAMPLES, WINDOW, MIN_GAP = 256, 64, 16  # v2: covers n>=144 tokens (v1's 150/50 masked 75% of docs as NaN)
NEW_NAMES = [f"s256_{i:03d}" for i in range(SAMPLES)]


def matrix(texts) -> np.ndarray:
    rows = []
    for t in texts:
        s = random_change_series(str(t), window=WINDOW, samples=SAMPLES,
                                 min_gap=MIN_GAP, metric="ck2", unit="tokens")
        if len(s) < SAMPLES:
            s = s + [np.nan] * (SAMPLES - len(s))
        rows.append(s)
    return np.array(rows, dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if not n.startswith("s256_")]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + NEW_NAMES)
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
