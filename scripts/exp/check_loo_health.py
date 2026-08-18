"""Check LOO health: bank-member A rows must NOT self-match (min > 0)."""
import numpy as np

d = np.load("data/derived/full_features.npz")
names = list(d["feature_names"])
i_min = names.index("ex_ai_min")
i_hmin = names.index("ex_hu_min")
Xa = d["X_A"]

zero_ai = int(np.sum(Xa[:, i_min] < 1e-12))
zero_hu = int(np.sum(Xa[:, i_hmin] < 1e-12))
print(f"A rows with ex_ai_min ~ 0: {zero_ai} (expected ~0 if LOO works)")
print(f"A rows with ex_hu_min ~ 0: {zero_hu} (expected ~0 if LOO works)")

# also check the restored trio's value sanity on A vs C
for n in ("ex_ai_mean_raw", "ex_contrast_min", "ex_contrast_mean",
          "bg_er", "cov2_ai", "dct_paircos_p25"):
    j = names.index(n)
    print(f"{n}: A med {np.nanmedian(d['X_A'][:, j]):.4f} | "
          f"C med {np.nanmedian(d['X_C'][:, j]):.4f}")
