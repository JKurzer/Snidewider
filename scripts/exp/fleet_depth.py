"""FLEET Z — HGB depth sweep (Donk: 'slightly deeper'). Train A, read C."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc, tpr_at_fpr

OUT = "docs/exp/fleet_depth.md"
ARMS = {
    "d4/i300 (current)": dict(max_depth=4, max_iter=300),
    "d6/i300": dict(max_depth=6, max_iter=300),
    "d6/i600": dict(max_depth=6, max_iter=600),
    "d8/i600": dict(max_depth=8, max_iter=600),
}


def main() -> None:
    d = np.load("data/derived/full_features.npz")
    X = {b: d[f"X_{b}"].astype(float) for b in "ABC"}
    y = {b: d[f"y_{b}"] for b in "ABC"}
    means = np.nan_to_num(np.nanmean(X["A"], axis=0))
    for b in "ABC":
        bad = np.where(~np.isfinite(X[b]))
        X[b][bad] = np.take(means, bad[1])

    lines = ["# FLEET Z — HGB depth sweep (train A, read C)\n\n",
             "| arm | AUROC C | TPR@1e-2 C | TPR@1e-3 C |\n|---|---|---|---|\n"]
    for arm, params in ARMS.items():
        m = HistGradientBoostingClassifier(learning_rate=0.08, max_features=0.5,
                                           random_state=7, **params).fit(X["A"], y["A"])
        s = m.predict_proba(X["C"])[:, 1]
        roc = auroc(list(s[y["C"] == 1]), list(s[y["C"] == 0]))
        r1 = tpr_at_fpr(list(s[y["C"] == 1]), list(s[y["C"] == 0]), fpr=1e-2)
        r3 = tpr_at_fpr(list(s[y["C"] == 1]), list(s[y["C"] == 0]), fpr=1e-3)
        lines.append(f"| {arm} | {roc:.4f} | {r1['tpr']:.3f} "
                     f"[{r1['tpr_lo']:.3f},{r1['tpr_hi']:.3f}] | {r3['tpr']:.3f} |\n")
        print(lines[-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
