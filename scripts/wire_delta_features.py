"""Wire the delta family (13 feats) into both caches (237 -> 250).

Rematch verdict (fleet S): +0.016 TPR@1e-2, +0.060 TPR@1e-3 over the 237
panel; delta13 alone 0.855/0.304. The 153-era tie verdict is overturned.
Idempotent: strips delta_/wdelta_ columns, appends fresh.
Usage: .venv\\Scripts\\python scripts\\wire_delta_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "exp"))
from fleet_condensates import DELTA_NAMES, delta_feats  # noqa: E402

from ai_text_detection.evaldata import split_buckets  # noqa: E402

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"


def matrix(texts) -> np.ndarray:
    return np.array([[delta_feats(str(t))[k] for k in DELTA_NAMES]
                     for t in texts], dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"]
             if not n.startswith(("delta_", "wdelta_"))]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + DELTA_NAMES)
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
