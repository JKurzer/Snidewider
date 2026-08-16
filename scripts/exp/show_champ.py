"""Print the current L0 champion's full feature list (selection on B @1e-2)."""
import numpy as np

from ai_text_detection import _l0learn_native
from ai_text_detection.metrics import auroc, tpr_at_fpr

data = np.load("data/derived/full_features.npz")
names = list(data["feature_names"])
X = {b: data[f"X_{b}"].astype(float) for b in "AB"}
y = {b: data[f"y_{b}"].astype(float) for b in "AB"}
means = np.nan_to_num(np.nanmean(X["A"], axis=0))
for b in "AB":
    bad = np.where(~np.isfinite(X[b]))
    X[b][bad] = np.take(means, bad[1])

best = None
for penalty in ("L0", "L0L2"):
    out = _l0learn_native.fit(np.asfortranarray(X["A"]), y["A"], penalty=penalty,
                              n_lambda=100, max_nnz=25)
    off = 0
    for Bm in out["betas"]:
        Bm = np.asarray(Bm)
        for j in range(Bm.shape[1]):
            beta, b0 = Bm[:, j], float(out["intercepts"][off + j])
            sb = 1 / (1 + np.exp(-(X["B"] @ beta + b0)))
            r = tpr_at_fpr(list(sb[y["B"] == 1]), list(sb[y["B"] == 0]), fpr=1e-2)
            key = (r["tpr"], auroc(list(sb[y["B"] == 1]), list(sb[y["B"] == 0])))
            if best is None or key > best[0]:
                best = (key, penalty, beta)
        off += Bm.shape[1]

(tpr_b, roc_b), penalty, beta = best
nz = np.nonzero(beta)[0]
print(f"champion: {penalty}, nnz={len(nz)}, B TPR@1e-2 {tpr_b:.3f}, B AUROC {roc_b:.3f}")
for i in nz:
    print(f"  {names[i]:<28} {beta[i]:+.4f}")
