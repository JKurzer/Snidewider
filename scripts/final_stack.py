"""Final stack baseline: 4 separated detectors (cached scores) -> meta on B -> C once.

Reads data/derived/base_scores.npz (regenerate with cache_base_scores.py).
Usage: .venv\\Scripts\\python scripts/final_stack.py
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_text_detection.metrics import auroc, tpr_at_fpr

DETECTORS = ("relative-burst", "qgram12", "exemplar", "dct-nobase")


def main() -> None:
    data = np.load("data/derived/base_scores.npz")
    labels = {b: data[f"labels_{b}"] for b in ("A", "B", "C")}

    print("== single detectors on C ==")
    for det in DETECTORS:
        s = data[f"{det}_C"]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        res = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        print(
            f"  {det:<15} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} "
            f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]"
        )

    Zb = np.column_stack([data[f"{det}_B"] for det in DETECTORS])
    Zc = np.column_stack([data[f"{det}_C"] for det in DETECTORS])
    metas = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(random_state=7),
        "mean-vote": None,
    }
    print("\n== stacked (meta on B, evaluated once on C) ==")
    for name, model in metas.items():
        s = Zc.mean(axis=1) if model is None else model.fit(Zb, labels["B"]).predict_proba(Zc)[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        res = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        print(
            f"  {name:<15} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} "
            f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]"
        )


if __name__ == "__main__":
    main()
