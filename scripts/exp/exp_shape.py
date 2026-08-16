"""Panel experiment: shape-map CK2 features (skeleton + dct_run) as detectors.

Solo AUROC/TPR for the shape set, then stacked with the cached 4 detectors.
Usage: .venv\\Scripts\\python scripts/exp_shape.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr
from ai_text_detection.shape import SHAPE_FEATURE_NAMES, shape_features

DETS = ("relative-burst", "qgram12", "exemplar", "dct-nobase")


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    cached = np.load("data/derived/base_scores.npz")

    feats, labels = {}, {}
    for name, sub in buckets.items():
        labels[name] = (sub.model != "human").to_numpy(int)
        rows = [[shape_features(str(t))[k] for k in SHAPE_FEATURE_NAMES] for t in sub.generation]
        X = np.array(rows, dtype=float)
        col_means = np.nanmean(X, axis=0)
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(col_means, bad[1])
        feats[name] = X
        print(f"  {name} featurized")

    print("\nper-feature AUROC (C):")
    for i, feat in enumerate(SHAPE_FEATURE_NAMES):
        s = feats["C"][:, i]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        print(f"  {feat:<24} {roc:.3f}")

    model = HistGradientBoostingClassifier(random_state=7).fit(feats["A"], labels["A"])
    shape_c = model.predict_proba(feats["C"])[:, 1]
    roc = auroc(list(shape_c[labels["C"] == 1]), list(shape_c[labels["C"] == 0]))
    res = tpr_at_fpr(list(shape_c[labels["C"] == 1]), list(shape_c[labels["C"] == 0]))
    print(f"\nshape solo on C: AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")

    shape_b = model.predict_proba(feats["B"])[:, 1]
    Zb = np.column_stack([cached[f"{d}_B"] for d in DETS] + [shape_b])
    Zc = np.column_stack([cached[f"{d}_C"] for d in DETS] + [shape_c])
    meta = HistGradientBoostingClassifier(random_state=7).fit(Zb, labels["B"])
    s = meta.predict_proba(Zc)[:, 1]
    roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
    res = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
    print(f"5-detector stack (hgb): AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")
    print("(4-detector reference: hgb AUROC 0.939 | TPR@1e-3 0.241)")


if __name__ == "__main__":
    main()
