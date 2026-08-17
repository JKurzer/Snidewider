"""Wire raw CSA measures into both caches (153 -> 156).

Donk's directive: these are RAW FEATURES for limited learners, not measures
to editorialize. No thresholds, no length control, no ratios. Just:
  csa_n         doc length in bytes (raw)
  csa_wt_rate   csa_wt<wt_huff> size_in_bytes / n
  csa_sada_rate csa_sada size_in_bytes / n

Idempotent: re-running replaces the csa_ columns. Usage:
.venv\\Scripts\\python scripts\\wire_csa_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection import _csa_native
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
NEW_NAMES = ["csa_n", "csa_wt_rate", "csa_sada_rate"]


def csa_matrix(texts) -> np.ndarray:
    rows = []
    for t in texts:
        b = str(t).encode("utf-8")
        out = _csa_native.csa_stats(b)
        n = max(1, len(b))
        rows.append([float(len(b)), out["csa_wt_bytes"] / n, out["csa_sada_bytes"] / n])
    return np.array(rows, dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if not n.startswith("csa_")]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + NEW_NAMES)
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features", flush=True)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    dev_new = {}
    for b in "ABC":
        dev_new[f"X_{b}"] = csa_matrix(buckets[b].generation)
        print(f"dev {b}: {dev_new[f'X_{b}'].shape}", flush=True)
    rewire(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_new = {
        "X_hu": csa_matrix(hold[hold.model == "human"].generation),
        "X_ai": csa_matrix(hold[hold.model != "human"].sample(n=20_000, random_state=97)
                           .generation),
    }
    for k, m in hold_new.items():
        print(f"holdout {k}: {m.shape}", flush=True)
    rewire(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
