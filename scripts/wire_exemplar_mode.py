"""Feature-set surgery per knockout verdict (Donk's orders):

DROP: ex_ai_mean_raw, ex_contrast_min, ex_contrast_mean, bg_er, cov2_ai,
dct_paircos_p25 (knockout diluters). ADD: ex_contrast_mode. 226 -> 221.

Idempotent. Usage: .venv\\Scripts\\python scripts\\wire_exemplar_mode.py
"""

import numpy as np
import pandas as pd

from ai_text_detection import qgram
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import (
    ExemplarBank,
    bank_self_indices,
    exemplar_features,
)

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
DROP = {"ex_ai_mean_raw", "ex_contrast_min", "ex_contrast_mean",
        "bg_er", "cov2_ai", "dct_paircos_p25"}
NEW = "ex_contrast_mode"
N_BANK = 150


def mode_col(texts, models, bank_ai, bank_hu, n_bank: int | None):
    rows = []
    ai_s = hu_s = None
    if n_bank is not None:
        ai_s, hu_s = bank_self_indices([str(m) for m in models], n_bank)
    for i, t in enumerate(texts):
        prof = qgram.profile(str(t).encode("utf-8"), 3)
        if ai_s is None:
            feats = exemplar_features(prof, bank_ai, bank_hu)
        else:
            feats = exemplar_features(prof, bank_ai, bank_hu, ai_s[i], hu_s[i])
        rows.append(feats[NEW])
    return np.array(rows, dtype=float)


def surgery(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    old = list(cache["feature_names"])
    names = [n for n in old if n not in DROP and n != NEW]
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

    dev_new = {}
    for b in "ABC":
        dev_new[f"X_{b}"] = mode_col(
            buckets[b].generation, buckets[b].model, bank_ai, bank_hu,
            N_BANK if b == "A" else None)
        print(f"dev {b} done", flush=True)
    surgery(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_new = {
        "X_hu": mode_col(hold[hold.model == "human"].generation, None,
                         bank_ai, bank_hu, None),
        "X_ai": mode_col(hold[hold.model != "human"]
                         .sample(n=20_000, random_state=97).generation, None,
                         bank_ai, bank_hu, None),
    }
    surgery(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)


if __name__ == "__main__":
    main()
