"""Final adjustment per Donk (2026-08-18): drop ex_contrast_mean,
dct_arc_cos, and ALL tg3_* trigrams; add ex_contrast_centroid (direct,
non-aggregate). shape_skeleton_step_mean STAYS (was doing something).
269 -> 236. Idempotent.
"""

import numpy as np
import pandas as pd

from ai_text_detection import qgram
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import (
    ExemplarBank,
    centroid_contrast,
    centroid_profile,
)

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
DROP = {"ex_contrast_mean", "dct_arc_cos"}
NEW = "ex_contrast_centroid"
N_BANK = 150


def col(texts, c_ai, c_hu) -> np.ndarray:
    return np.array([centroid_contrast(qgram.profile(str(t).encode("utf-8"), 3),
                                       c_ai, c_hu)
                     for t in texts], dtype=float)


def surgery(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    old = list(cache["feature_names"])
    names = [n for n in old if n not in DROP and n != NEW
             and not n.startswith("tg3_")]
    keep_idx = [old.index(n) for n in names]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, keep_idx], new_cols[key]])
    store["feature_names"] = np.array(names + [NEW])
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features", flush=True)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    a = buckets["A"]
    bank_ai = ExemplarBank.from_texts(
        [str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts(
        [str(t) for t in a[a.model == "human"].generation[:N_BANK]])
    c_ai, c_hu = centroid_profile(bank_ai), centroid_profile(bank_hu)
    print("centroids built", flush=True)

    dev_new = {f"X_{b}": col(buckets[b].generation, c_ai, c_hu) for b in "ABC"}
    for b in "ABC":
        print(f"dev {b} done", flush=True)
    surgery(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_new = {
        "X_hu": col(hold[hold.model == "human"].generation, c_ai, c_hu),
        "X_ai": col(hold[hold.model != "human"].sample(n=20_000, random_state=97)
                    .generation, c_ai, c_hu),
    }
    surgery(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
