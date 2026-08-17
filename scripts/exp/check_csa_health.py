"""CSA health check: cache columns vs probe distribution + redundancy audit."""
import numpy as np
import pandas as pd

from ai_text_detection.evaldata import split_buckets

d = np.load("data/derived/full_features.npz")
names = list(d["feature_names"])
i_wt = names.index("csa_wt_rate")
i_sada = names.index("csa_sada_rate")
i_z = names.index("stat_zlib_ratio")
i_bz = names.index("stat_bz2_ratio")

df = pd.read_parquet("data/derived/raid_splits.parquet")
c = split_buckets(df)["C"]
X = d["X_C"]
y = d["y_C"]
n_tok = np.array([len(str(t).encode()) for t in c.generation])

print("csa_wt_rate in cache: class x length band (median)")
for lo, hi in ((200, 1000), (1000, 2000), (2000, 4000)):
    m = (n_tok >= lo) & (n_tok < hi)
    for cls, lab in ((1, "ai"), (0, "hu")):
        v = X[(y == cls) & m, i_wt]
        v = v[np.isfinite(v)]
        if len(v) > 10:
            print(f"  {lo}-{hi} {lab}: med {np.median(v):.3f} p95 {np.percentile(v, 95):.3f} n={len(v)}")


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


print(f"\nspearman(csa_wt_rate, stat_zlib_ratio): {spearman(X[:, i_wt], X[:, i_z]):.3f}")
print(f"spearman(csa_sada_rate, stat_zlib_ratio): {spearman(X[:, i_sada], X[:, i_z]):.3f}")
print(f"spearman(csa_wt_rate, stat_bz2_ratio): {spearman(X[:, i_wt], X[:, i_bz]):.3f}")
print(f"spearman(csa_wt_rate, csa_n): {spearman(X[:, i_wt], X[:, names.index('csa_n')]):.3f}")
