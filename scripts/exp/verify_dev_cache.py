"""One-shot idempotency check: dev cache after re-wire vs .bak (must be identical)."""
import numpy as np

old = np.load("data/derived/full_features.npz.bak")
new = np.load("data/derived/full_features.npz")
assert list(old["feature_names"]) == list(new["feature_names"]), "feature order changed!"
for k in ("X_A", "X_B", "X_C", "y_A", "y_B", "y_C"):
    a, b = old[k], new[k]
    same = (a.shape == b.shape) and np.array_equal(np.nan_to_num(a), np.nan_to_num(b))
    print(f"{k}: {a.shape} vs {b.shape} identical={same}")
    assert same, f"{k} differs!"
print("IDEMPOTENT: dev cache bit-identical after re-wire")
