"""Restore cv_cv_rate + chr_para_len_mean (Donk: revert the drops; holdout
said they were earning - the knockout's tail column misread again).
251 -> 253, exact original positions. Idempotent.
Usage: .venv\\Scripts\\python scripts\\restore_drops.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.chargrams import chargram_features
from ai_text_detection.charstat import charstat_features
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
# name -> (module value function, insert-before anchor)
SPECS = {
    "chr_para_len_mean": (lambda t: charstat_features(t)["para_len_mean"],
                          "chr_para_len_stdev"),
    "cv_cv_rate": (lambda t: chargram_features(t)["cv_cv_rate"], "cv_vc_rate"),
}


def restore(path: str, keys: list[str], cols: dict[str, dict[str, np.ndarray]]) -> None:
    cache = np.load(path)
    names = list(cache["feature_names"])
    for n, (_, anchor) in SPECS.items():
        if n in names:
            names.remove(n)  # idempotent: re-insert fresh below
        pos = names.index(anchor)
        names = names[:pos] + [n] + names[pos:]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        base = [n for n in cache["feature_names"]]
        X = store[key]
        blocks = []
        for n in names:
            if n in SPECS:
                blocks.append(cols[key][n][:, None])
            else:
                j = base.index(n)
                blocks.append(X[:, j:j + 1])
        store[key] = np.hstack(blocks)
    store["feature_names"] = np.array(names)
    np.savez(path, **store)
    print(f"{path}: -> {len(names)} features", flush=True)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    dev_cols = {}
    for b in "ABC":
        texts = [str(t) for t in buckets[b].generation]
        dev_cols[f"X_{b}"] = {
            n: np.array([fn(t) for t in texts], dtype=float)
            for n, (fn, _) in SPECS.items()
        }
        print(f"dev {b} done", flush=True)
    restore(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_cols)

    hold = df[df.fold == "holdout"]
    hold_cols = {}
    for key, sub in (("X_hu", hold[hold.model == "human"]),
                     ("X_ai", hold[hold.model != "human"]
                      .sample(n=20_000, random_state=97))):
        texts = [str(t) for t in sub.generation]
        hold_cols[key] = {n: np.array([fn(t) for t in texts], dtype=float)
                          for n, (fn, _) in SPECS.items()}
    restore(HOLD_NPZ, ["X_hu", "X_ai"], hold_cols)


if __name__ == "__main__":
    main()
