"""Small boosted forest over the full 81-feature panel.

HistGradientBoosting (boosted trees with feature subsampling = the 'random'
in the forest), trained on bucket A, feature importances on B, final numbers
read ONCE on C. Comparison anchor: the 4-detector stacked HGB meta
(AUROC 0.939 / TPR@1e-3 0.241 from final_stack.py).

Usage: .venv\\Scripts\\python scripts/train_forest.py
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ai_text_detection.metrics import auroc, tpr_at_fpr


def main() -> None:
    data = np.load("data/derived/full_features.npz")
    names = list(data["feature_names"])
    Xa, ya = data["X_A"], data["y_A"]
    Xb, yb = data["X_B"], data["y_B"]
    Xc, yc = data["X_C"], data["y_C"]
    for X in (Xa, Xb, Xc):
        col_means = np.nanmean(Xa, axis=0)  # A-fit imputation, applied to all
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(col_means, bad[1])
    print(f"train A {Xa.shape}, meta B {Xb.shape}, test C {Xc.shape}, {len(names)} features")

    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.08, max_features=0.5, random_state=7
    )
    model.fit(Xa, ya)

    for bucket, X, y in (("B", Xb, yb), ("C", Xc, yc)):
        s = model.predict_proba(X)[:, 1]
        roc = auroc(list(s[y == 1]), list(s[y == 0]))
        res = tpr_at_fpr(list(s[y == 1]), list(s[y == 0]))
        tag = "FINAL" if bucket == "C" else "select"
        print(f"{bucket} ({tag}): AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} "
              f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")

    imp = permutation_importance(model, Xb, yb, n_repeats=5, random_state=7, n_jobs=-1)
    order = np.argsort(-imp.importances_mean)
    print("\ntop 15 features by permutation importance (on B):")
    for i in order[:15]:
        print(f"  {names[i]:<28} {imp.importances_mean[i]:+.4f}")


if __name__ == "__main__":
    main()
