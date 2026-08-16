"""RAID-paper-comparable numbers: TPR at FPR=5% for our detectors + stacks.

RAID's leaderboard reports accuracy at calibrated FPR=5% (Tables 5-6).
This computes ours from the cached C-bucket scores. Usage: python scripts/compare_sota.py
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_text_detection.metrics import tpr_at_fpr

DETS = ("relative-burst", "qgram12", "exemplar", "dct-nobase")


def tpr5(s_ai, s_hu) -> float:
    return tpr_at_fpr(list(s_ai), list(s_hu), 0.05)["tpr"]


def main() -> None:
    d = np.load("data/derived/base_scores.npz")
    yb, yc = d["labels_B"], d["labels_C"]
    print(f"{'model':<16} {'TPR@FPR=5%':>10} {'TPR@FPR=0.1%':>12}")
    for det in DETS:
        s = d[f"{det}_C"]
        print(f"{det:<16} {tpr5(s[yc == 1], s[yc == 0]):>10.3f} {tpr_at_fpr(list(s[yc==1]), list(s[yc==0]))['tpr']:>12.3f}")
    Zb = np.column_stack([d[f"{det}_B"] for det in DETS])
    Zc = np.column_stack([d[f"{det}_C"] for det in DETS])
    for name, m in (
        ("stack-logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("stack-hgb", HistGradientBoostingClassifier(random_state=7)),
    ):
        m.fit(Zb, yb)
        s = m.predict_proba(Zc)[:, 1]
        print(f"{name:<16} {tpr5(s[yc == 1], s[yc == 0]):>10.3f} {tpr_at_fpr(list(s[yc==1]), list(s[yc==0]))['tpr']:>12.3f}")


if __name__ == "__main__":
    main()
