"""Post-restore sanity: retrain on A, read C (original 226 baseline: 0.9930/0.904/0.763)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)

d = np.load("data/derived/full_features.npz")
names = list(d["feature_names"])
print("first 35 names:", names[:35])
X = {b: d[f"X_{b}"].astype(float) for b in "ABC"}
y = {b: d[f"y_{b}"] for b in "ABC"}
means = np.nan_to_num(np.nanmean(X["A"], axis=0))
for b in "ABC":
    bad = np.where(~np.isfinite(X[b]))
    X[b][bad] = np.take(means, bad[1])

m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"], y["A"])
s = m.predict_proba(X["C"])[:, 1]
roc = auroc(list(s[y["C"] == 1]), list(s[y["C"] == 0]))
t1 = tpr_at_fpr(list(s[y["C"] == 1]), list(s[y["C"] == 0]), fpr=1e-2)["tpr"]
t3 = tpr_at_fpr(list(s[y["C"] == 1]), list(s[y["C"] == 0]), fpr=1e-3)["tpr"]
print(f"restored-226 dev read: AUROC {roc:.4f} TPR@1e-2 {t1:.3f} TPR@1e-3 {t3:.3f}")
print("(original 226 baseline:  AUROC 0.9930 TPR@1e-2 0.904 TPR@1e-3 0.763)")
