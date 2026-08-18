"""Wire the chargram + BWT/Wheeler pack into both caches (226 -> 268).

Fleet Q: +0.010 TPR@1e-2 increment over 226 (0.914 vs 0.904); top solos are
the sdsl-backed BWT stats (bwt_char_entropy 0.604, bwt_run_max tail 0.080)
and initial_char_entropy (tail 0.114). Column order: chargrams then bwt.

Idempotent: strips tg3_/cv_/initial_/bwt_ columns, appends fresh.
Usage: .venv\\Scripts\\python scripts\\wire_chargram_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.bwt_stats import BWT_FEATURE_NAMES, bwt_features
from ai_text_detection.chargrams import CHARGRAM_FEATURE_NAMES, chargram_features
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
PREFIXES = ("tg3_", "cv_", "bwt_")
NEW_NAMES = list(CHARGRAM_FEATURE_NAMES) + list(BWT_FEATURE_NAMES)


def matrix(texts) -> np.ndarray:
    rows = []
    for t in texts:
        text = str(t)
        cg = chargram_features(text)
        bw = bwt_features(text)
        rows.append([cg[k] for k in CHARGRAM_FEATURE_NAMES]
                    + [bw[k] for k in BWT_FEATURE_NAMES])
    return np.array(rows, dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"]
             if not n.startswith(PREFIXES) and n != "initial_char_entropy"]
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
