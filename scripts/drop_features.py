"""Drop named feature columns from both caches (Donk-ordered knockout cuts).

Usage: .venv\\Scripts\\python scripts\\drop_features.py cv_cv_rate chr_para_len_mean
"""

import sys

import numpy as np

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"


def drop(path: str, keys: list[str], victims: list[str]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if n not in victims]
    keep_idx = [list(cache["feature_names"]).index(n) for n in names]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = store[key][:, keep_idx]
    store["feature_names"] = np.array(names)
    np.savez(path, **store)
    print(f"{path}: -> {len(names)} features", flush=True)


if __name__ == "__main__":
    victims = sys.argv[1:]
    if not victims:
        raise SystemExit("name the features to drop")
    drop(DEV_NPZ, ["X_A", "X_B", "X_C"], victims)
    drop(HOLD_NPZ, ["X_hu", "X_ai"], victims)
