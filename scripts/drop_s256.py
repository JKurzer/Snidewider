"""Strip the raw256 block (s256_*) from both caches: 509 -> 253.

The holdout verdict (twice, at both coverages): the raw256 vector does not
earn (v1 flat, v2 dilutes). RULE 6. The other 253 columns are untouched.
Usage: .venv\\Scripts\\python scripts\\drop_s256.py
"""

import numpy as np

for path, keys in (("data/derived/full_features.npz", ["X_A", "X_B", "X_C"]),
                   ("data/derived/holdout_features.npz", ["X_hu", "X_ai"])):
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if not n.startswith("s256_")]
    keep = [list(cache["feature_names"]).index(n) for n in names]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = store[key][:, keep]
    store["feature_names"] = np.array(names)
    np.savez(path, **store)
    print(f"{path}: -> {len(names)} features", flush=True)
