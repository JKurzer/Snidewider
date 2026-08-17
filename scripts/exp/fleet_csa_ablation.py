"""CSA ablation: does the CSA trio (62% of pipeline cost) earn its keep?

Arms (tuned HGB, train A, read C): full 157 vs 157 minus {csa_n, csa_wt_rate,
csa_sada_rate}. Same protocol as every other feature decision in this repo.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)

d = np.load("data/derived/full_features.npz")
names = list(d["feature_names"])
csa_cols = [i for i, n in enumerate(names) if n.startswith("csa_")]
keep = [i for i in range(len(names)) if i not in csa_cols]
X = {b: d[f"X_{b}"].astype(float) for b in "ABC"}
y = {b: d[f"y_{b}"] for b in "ABC"}
means = np.nan_to_num(np.nanmean(X["A"], axis=0))
for b in "ABC":
    bad = np.where(~np.isfinite(X[b]))
    X[b][bad] = np.take(means, bad[1])

for arm, cols in (("full157", list(range(157))), ("no-csa154", keep)):
    m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"][:, cols], y["A"])
    s = m.predict_proba(X["C"][:, cols])[:, 1]
    roc = auroc(list(s[y["C"] == 1]), list(s[y["C"] == 0]))
    for fpr in (1e-2, 1e-3):
        r = tpr_at_fpr(list(s[y["C"] == 1]), list(s[y["C"] == 0]), fpr=fpr)
        print(f"{arm:<10} C AUROC {roc:.4f} TPR@{fpr:.0e} {r['tpr']:.3f} "
              f"[{r['tpr_lo']:.3f},{r['tpr_hi']:.3f}]", flush=True)
