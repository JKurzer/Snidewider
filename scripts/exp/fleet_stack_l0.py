"""FLEET G — the old stacked design: HGB score as a feature for stage-2.

Split decision tonight: L0 champ owns 1e-2/5e-2, HGB owns 1e-3. The classic
fix: hand the HGB score to the next model as a feature and let IT arbitrate.

Cross-fit hygiene (no in-sample scores anywhere):
  HGB_A = tuned HGB trained on A(120)  -> scores B, C, holdout (honest)
  HGB_B = same, trained on B           -> scores A (honest)
  stage-2 trains on B [X, s_A], L0 path point selected on A [X, s_B],
  C read once per arm; holdout read for both arms (pre-declared pair).

Arms: L0/L0L2 path over 121 feats; stage-2 HGB over 121 feats.
Reference (committed tonight): L0 alone dev C 0.919/0.597, holdout
0.9103/0.536; HGB alone holdout 0.9061/0.465.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_stack_l0.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import _l0learn_native
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_stack_l0.md"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    data = np.load("data/derived/full_features.npz")
    names = list(data["feature_names"]) + ["hgb_score"]
    X = {b: data[f"X_{b}"].astype(float) for b in "ABC"}
    y = {b: data[f"y_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(X["A"], axis=0))
    for b in "ABC":
        bad = np.where(~np.isfinite(X[b]))
        X[b][bad] = np.take(means, bad[1])

    hgb_a = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"], y["A"])
    hgb_b = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["B"], y["B"])
    s = {"A": hgb_b.predict_proba(X["A"])[:, 1], "B": hgb_a.predict_proba(X["B"])[:, 1],
         "C": hgb_a.predict_proba(X["C"])[:, 1]}
    Z = {b: np.column_stack([X[b], s[b]]) for b in "ABC"}

    def metrics(scores, yb):
        roc = auroc(list(scores[yb == 1]), list(scores[yb == 0]))
        r = tpr_at_fpr(list(scores[yb == 1]), list(scores[yb == 0]), fpr=1e-2)
        return roc, r["tpr"]

    # --- arm 1: L0/L0L2 path over the 121-feat stacked matrix ---
    best = None
    for penalty in ("L0", "L0L2"):
        out = _l0learn_native.fit(np.asfortranarray(Z["B"]), y["B"], penalty=penalty,
                                  n_lambda=100, max_nnz=25)
        off = 0
        for Bm in out["betas"]:
            Bm = np.asarray(Bm)
            for j in range(Bm.shape[1]):
                beta, b0 = Bm[:, j], float(out["intercepts"][off + j])
                sa = sigmoid(Z["A"] @ beta + b0)  # select on A
                key = (tpr_at_fpr(list(sa[y["A"] == 1]), list(sa[y["A"] == 0]), fpr=1e-2)["tpr"],
                       auroc(list(sa[y["A"] == 1]), list(sa[y["A"] == 0])))
                if best is None or key > best[0]:
                    best = (key, penalty, beta, b0)
            off += Bm.shape[1]
    (tpr_a, roc_a), pen, beta, b0 = best
    roc_c, tpr_c = metrics(sigmoid(Z["C"] @ beta + b0), y["C"])
    picked = [(names[i], beta[i]) for i in np.nonzero(beta)[0]]
    print(f"arm1 L0-stack: {pen} nnz={len(picked)} | select A {roc_a:.3f}/{tpr_a:.3f} "
          f"| C {roc_c:.3f}/{tpr_c:.3f}", flush=True)
    print("  hgb_score weight:", dict((n, w) for n, w in picked).get("hgb_score", "NOT PICKED"))

    # --- arm 2: stage-2 HGB over the stacked matrix ---
    hgb2 = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Z["B"], y["B"])
    roc_c2, tpr_c2 = metrics(hgb2.predict_proba(Z["C"])[:, 1], y["C"])
    print(f"arm2 HGB-stack: C {roc_c2:.3f}/{tpr_c2:.3f}", flush=True)

    # --- holdout for both arms (pre-declared pair) ---
    hold = np.load("data/derived/holdout_features.npz")
    hnames = list(hold["feature_names"])
    idx = [hnames.index(n) for n in names[:-1]]
    Xh = {k: hold[k][:, idx].astype(float) for k in ("X_hu", "X_ai")}
    for k in Xh:
        bad = np.where(~np.isfinite(Xh[k]))
        Xh[k][bad] = np.take(means, bad[1])
    Zh = {k: np.column_stack([Xh[k], hgb_a.predict_proba(Xh[k])[:, 1]]) for k in Xh}
    s1 = sigmoid(Zh["X_ai"] @ beta + b0)
    s1h = sigmoid(Zh["X_hu"] @ beta + b0)
    s2 = hgb2.predict_proba(Zh["X_ai"][:, :])[:, 1]
    s2h = hgb2.predict_proba(Zh["X_hu"][:, :])[:, 1]

    lines = ["# FLEET G — stacked HGB-score feature\n\n",
             f"arm1 L0-stack ({pen}, nnz={len(picked)}): C AUROC {roc_c:.3f} TPR@1e-2 {tpr_c:.3f}; "
             f"hgb_score weight {dict((n, w) for n, w in picked).get('hgb_score', 'NOT PICKED')}\n",
             f"arm2 HGB-stack: C AUROC {roc_c2:.3f} TPR@1e-2 {tpr_c2:.3f}\n\n",
             "## holdout\n\n"]
    for tag, ai, hu in (("L0-stack", s1, s1h), ("HGB-stack", s2, s2h)):
        roc = auroc(list(ai), list(hu))
        cells = []
        for fpr in (5e-2, 1e-2, 1e-3):
            r = tpr_at_fpr(list(ai), list(hu), fpr=fpr)
            cells.append(f"{r['tpr']:.3f} [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}]")
        lines.append(f"- {tag}: AUROC {roc:.4f} | " + " | ".join(cells) + "\n")
        print(lines[-1], flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
