"""Wire the fleet-A2 champion stat into both caches (156 -> 157).

qg_s256_ck2_mean: W=150-token random pairs, CK2 metric, samples=256,
min_gap=50 (docs/exp/fleet_qgmid2.md + holdout confirmation). Earned its
seat in the increment trial: +0.029 TPR@1e-3 on dev C with AUROC flat.

Idempotent: re-running replaces the column. Usage:
.venv\\Scripts\\python scripts\\wire_s256_feature.py
"""

import numpy as np
import pandas as pd

from ai_text_detection import burst
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
NAME = "qg_s256_ck2_mean"


def col(texts) -> np.ndarray:
    out = []
    for t in texts:
        s = burst.random_change_series(str(t), window=150, samples=256,
                                       min_gap=50, metric="ck2", unit="tokens")
        out.append(float(np.mean(s)) if s else np.nan)
    return np.array(out, dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if n != NAME]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + [NAME])
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
        "X_ai": col(hold[hold.model != "human"].sample(n=20_000, random_state=97).generation),
    }
    rewire(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
