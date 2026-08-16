"""Probe the cross-fit asymmetry: is bucket A intrinsically easy, or leaking?"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)

data = np.load("data/derived/full_features.npz")
X = {b: data[f"X_{b}"].astype(float) for b in "ABC"}
y = {b: data[f"y_{b}"].astype(int) for b in "ABC"}
means = np.nan_to_num(np.nanmean(X["A"], axis=0))
for b in "ABC":
    bad = np.where(~np.isfinite(X[b]))
    X[b][bad] = np.take(means, bad[1])

hgb_a = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"], y["A"])
hgb_b = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["B"], y["B"])


def roc(model, b):
    s = model.predict_proba(X[b])[:, 1]
    return auroc(list(s[y[b] == 1]), list(s[y[b] == 0]))


print(f"hgb_a -> A (in-sample):  {roc(hgb_a, 'A'):.4f}")
print(f"hgb_a -> B (honest):     {roc(hgb_a, 'B'):.4f}")
print(f"hgb_a -> C (honest):     {roc(hgb_a, 'C'):.4f}")
print(f"hgb_b -> B (in-sample):  {roc(hgb_b, 'B'):.4f}")
print(f"hgb_b -> A (honest):     {roc(hgb_b, 'A'):.4f}  <- the suspicious 1.000")
print(f"hgb_b -> C (honest):     {roc(hgb_b, 'C'):.4f}")

# feature-block ablation on the suspicious direction: hgb_b -> A
names = list(data["feature_names"])
fams = {"rel": (0, 8), "qg": (8, 20), "ex": (20, 31), "dct": (31, 81),
        "shape": (81, 89), "stat": (89, 111), "cov": (111, 120)}
for fam, (lo, hi) in fams.items():
    cols = [i for i in range(120) if not (lo <= i < hi)]
    h = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["B"][:, cols], y["B"])
    s = h.predict_proba(X["A"][:, cols])[:, 1]
    print(f"hgb_b -> A without {fam:<6}: {auroc(list(s[y['A']==1]), list(s[y['A']==0])):.4f}")
