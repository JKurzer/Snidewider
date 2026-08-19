"""Holdout read for the depth candidates (d6/i300, d8/i600) vs the d4 incumbent."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc, tpr_at_fpr

ARMS = {
    "d4/i300 (incumbent)": dict(max_depth=4, max_iter=300),
    "d6/i300": dict(max_depth=6, max_iter=300),
    "d8/i600": dict(max_depth=8, max_iter=600),
}


def main() -> None:
    dev = np.load("data/derived/full_features.npz")
    hold = np.load("data/derived/holdout_features.npz")
    dev_names = list(dev["feature_names"])
    hold_names = list(hold["feature_names"])
    col = {n: dev_names.index(n) for n in hold_names}
    cols = [col[n] for n in hold_names]
    means = np.nan_to_num(np.nanmean(dev["X_A"][:, cols], axis=0))

    def imp(X):
        X = X.copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(means, bad[1])
        return X

    Xa = imp(dev["X_A"][:, cols])
    ya = dev["y_A"]
    X_hu = imp(hold["X_hu"])
    X_ai = imp(hold["X_ai"])

    for arm, params in ARMS.items():
        hgb = HistGradientBoostingClassifier(learning_rate=0.08, max_features=0.5,
                                             random_state=7, **params).fit(Xa, ya)
        s_ai = hgb.predict_proba(X_ai)[:, 1]
        s_hu = hgb.predict_proba(X_hu)[:, 1]
        roc = auroc(list(s_ai), list(s_hu))
        cells = []
        for fpr in (5e-2, 1e-2, 1e-3):
            r = tpr_at_fpr(list(s_ai), list(s_hu), fpr=fpr)
            cells.append(f"{r['tpr']:.3f}")
        print(f"  {arm:<22} AUROC {roc:.4f} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
