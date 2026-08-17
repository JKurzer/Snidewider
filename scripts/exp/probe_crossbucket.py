"""Cross-bucket training matrix: is the 156-feat strength bucket-agnostic?

If distill36's power were an A-side artifact, reads would collapse when the
train bucket changes. Train HGB on each bucket, read the other two.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)

data = np.load("data/derived/full_features.npz")
X = {b: data[f"X_{b}"].astype(float) for b in "ABC"}
y = {b: data[f"y_{b}"].astype(int) for b in "ABC"}
means = np.nan_to_num(np.nanmean(X["A"], axis=0))
for b in "ABC":
    bad = np.where(~np.isfinite(X[b]))
    X[b][bad] = np.take(means, bad[1])

DISTILL = list(range(120, 156))

for train in "ABC":
    m_all = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X[train], y[train])
    m_dis = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X[train][:, DISTILL], y[train])
    for read in "ABC":
        if read == train:
            continue
        for tag, model, cols in (("full156", m_all, slice(None)),
                                 ("distill36", m_dis, DISTILL)):
            s = model.predict_proba(X[read][:, cols])[:, 1]
            roc = auroc(list(s[y[read] == 1]), list(s[y[read] == 0]))
            r = tpr_at_fpr(list(s[y[read] == 1]), list(s[y[read] == 0]), fpr=1e-2)
            print(f"train {train} -> read {read} [{tag:<9}] AUROC {roc:.3f} "
                  f"TPR@1e-2 {r['tpr']:.3f}", flush=True)
