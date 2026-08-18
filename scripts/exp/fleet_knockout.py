"""FLEET P — knockout:1: leave-one-out feature ablation over the 226 panel.

For each feature: drop it, retrain tuned HGB on A, read C. Delta vs the
full-226 baseline. Features whose removal costs nothing (or helps) are
noise/dilution candidates. ~226 trainings; runs a while.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_knockout.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_knockout.md"


def readout(model, X, y):
    s = model.predict_proba(X)[:, 1]
    roc = auroc(list(s[y == 1]), list(s[y == 0]))
    t1 = tpr_at_fpr(list(s[y == 1]), list(s[y == 0]), fpr=1e-2)["tpr"]
    t3 = tpr_at_fpr(list(s[y == 1]), list(s[y == 0]), fpr=1e-3)["tpr"]
    return roc, t1, t3


def main() -> None:
    d = np.load("data/derived/full_features.npz")
    names = list(d["feature_names"])
    n_feat = len(names)
    X = {b: d[f"X_{b}"].astype(float) for b in "ABC"}
    y = {b: d[f"y_{b}"] for b in "ABC"}
    means = np.nan_to_num(np.nanmean(X["A"], axis=0))
    for b in "ABC":
        bad = np.where(~np.isfinite(X[b]))
        X[b][bad] = np.take(means, bad[1])

    full = list(range(n_feat))
    base = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"], y["A"])
    b_roc, b_t1, b_t3 = readout(base, X["C"], y["C"])
    print(f"baseline (226): AUROC {b_roc:.4f} TPR@1e-2 {b_t1:.3f} TPR@1e-3 {b_t3:.3f}",
          flush=True)

    rows = []
    for j in range(n_feat):
        cols = [c for c in full if c != j]
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"][:, cols], y["A"])
        roc, t1, t3 = readout(m, X["C"][:, cols], y["C"])
        rows.append((names[j], roc - b_roc, t1 - b_t1, t3 - b_t3))
        if j % 25 == 0:
            print(f"{j}/{n_feat} knocked", flush=True)

    rows.sort(key=lambda r: r[1] + r[2] + r[3])  # most dilutive first (positive delta)
    lines = [f"# FLEET P — knockout:1 (baseline 226: AUROC {b_roc:.4f}, "
             f"TPR@1e-2 {b_t1:.3f}, TPR@1e-3 {b_t3:.3f})\n\n",
             "delta = knockout metric - baseline (POSITIVE = feature was diluting)\n\n",
             "| feature | dAUROC | dTPR@1e-2 | dTPR@1e-3 |\n|---|---|---|---|\n"]
    for name, dr, dt1, dt3 in rows:
        lines.append(f"| {name} | {dr:+.4f} | {dt1:+.3f} | {dt3:+.3f} |\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines[:25]))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
