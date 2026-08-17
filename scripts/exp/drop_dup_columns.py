"""One-shot: drop the exact-duplicate distill columns from both caches.

fleet I audit found rho=1.0 pairs; collapse.py shed col_spec_k1, col_spec_k2,
col_cond_entropy (they duplicated stat_hapax, stat_dis, chr_bigram_cond_entropy).
This drops those cache columns to match. 156 -> 153.
"""
import numpy as np

DROP = {"col_spec_k1", "col_spec_k2", "col_cond_entropy"}

for path in ("data/derived/full_features.npz", "data/derived/holdout_features.npz"):
    cache = np.load(path)
    names = list(cache["feature_names"])
    keep = [i for i, n in enumerate(names) if n not in DROP]
    store = {}
    for k in cache:
        if k.startswith("X_"):
            store[k] = cache[k][:, keep]
        elif k == "feature_names":
            store[k] = np.array([names[i] for i in keep])
        else:
            store[k] = cache[k]
    np.savez(path, **store)
    print(f"{path}: {len(names)} -> {len(keep)} features")
