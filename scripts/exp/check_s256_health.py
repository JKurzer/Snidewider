"""s256 series vital signs: value range, variance across docs, NaN rate,
per-column variance. The 'flat holdout' suspects: inert columns."""
import numpy as np
import pandas as pd

from ai_text_detection.burst import random_change_series
from ai_text_detection.evaldata import split_buckets

c = split_buckets(pd.read_parquet("data/derived/raid_splits.parquet"))["C"]
texts = [str(t) for t in c.generation[:200]]

rows = []
n_empty = 0
for t in texts:
    s = random_change_series(t, window=150, samples=256, min_gap=50,
                             metric="ck2", unit="tokens")
    if not s:
        n_empty += 1
        s = [np.nan] * 256
    elif len(s) < 256:
        s = s + [np.nan] * (256 - len(s))
    rows.append(s)
X = np.array(rows, dtype=float)

print(f"docs: {len(texts)}, empty-series docs: {n_empty}")
print(f"value range: min {np.nanmin(X):.4f} max {np.nanmax(X):.4f} "
      f"mean {np.nanmean(X):.4f}")
print(f"per-doc NaN fraction: mean {np.isnan(X).mean():.3f}")
col_var = np.nanvar(X, axis=0)
print(f"per-column variance: median {np.median(col_var):.6f} "
      f"max {col_var.max():.6f} min {col_var.min():.6f}")
print(f"columns with zero variance: {(col_var == 0).sum()}/256")
row_var = np.nanvar(X, axis=1)
print(f"per-doc (row) variance: median {np.median(row_var):.6f}")
# determinism
s1 = random_change_series(texts[0], 150, 256, 50, "ck2", "tokens")
s2 = random_change_series(texts[0], 150, 256, 50, "ck2", "tokens")
print(f"deterministic: {s1 == s2}")
