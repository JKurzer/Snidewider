"""Holdout FPR ladder — frozen models re-read at the field's operating points.

The final exam (34c3c46) read the holdout once at a C-set threshold (FPRa
0.00053). This script re-reads the SAME frozen models across the reporting
ladder from docs/sota-operating-points.md: 5e-2 (RAID field norm), 1e-2,
1e-3 (stretch; holdout has 11,371 humans => k=11, dev buckets can't do this).

Frozen = retrained deterministically (HGB rs=7, same buckets/code as
final_exam.py) and applied to the CACHED 81-feature holdout panel. Sanity
gate: holdout AUROC must reproduce the exam's 0.714. Thresholds here come
from the holdout score distribution itself (a ROC read, RAID-style), not
re-tuned model selection — no champions are picked on holdout.

L0 is OFFLINE (2026-08-17): the panel HGB beat the L0 champion on every
holdout rung, so the L0 arm was removed from this flow. It stays
resurrectable: the _l0learn_native extension ships and exp_l0_vs_hgb.py +
show_champ.py reproduce the champion. Rationale in the git log.

Usage: .venv\\Scripts\\python scripts\\exp\\holdout_fpr_ladder.py
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.metrics import auroc, tpr_at_fpr

FAMS = {"relative-burst": (0, 8), "qgram12": (8, 20), "exemplar": (20, 31), "dct-nobase": (31, 81)}
LADDER = (5e-2, 1e-2, 1e-3)
EXAM_AUROC = 0.714


def impute(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    X = X.copy()
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(means, bad[1])
    return X


def ladder(tag: str, s_ai: np.ndarray, s_hu: np.ndarray) -> None:
    roc = auroc(list(s_ai), list(s_hu))
    cells = []
    for fpr in LADDER:
        r = tpr_at_fpr(list(s_ai), list(s_hu), fpr=fpr)
        cells.append(f"TPR@{fpr:.0e} {r['tpr']:.3f} [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}]"
                     f" (FPRa {r['fpr_achieved']:.1e})")
    print(f"  {tag:<16} AUROC {roc:.4f} | " + " | ".join(cells))


def main() -> None:
    dev = np.load("data/derived/full_features.npz")
    hold = np.load("data/derived/holdout_features.npz")
    dev_names = list(dev["feature_names"])
    hold_names = list(hold["feature_names"])
    col = {n: dev_names.index(n) for n in hold_names}  # name-matched, order-proof

    cols = [col[n] for n in hold_names]  # dev columns matching the 81 holdout feats
    means = np.nan_to_num(np.nanmean(dev["X_A"][:, cols], axis=0))
    Xa = impute(dev["X_A"][:, cols], means)
    Xb = impute(dev["X_B"][:, cols], means)
    ya, yb = dev["y_A"], dev["y_B"]
    X_hu = impute(hold["X_hu"], means)
    X_ai = impute(hold["X_ai"], means)

    # --- frozen exam ensemble: 4 family HGBs on A, HGB meta on B ---
    dets = {f: HistGradientBoostingClassifier(random_state=7).fit(Xa[:, lo:hi], ya)
            for f, (lo, hi) in FAMS.items()}
    panel = lambda X: np.column_stack([dets[f].predict_proba(X[:, lo:hi])[:, 1]
                                       for f, (lo, hi) in FAMS.items()])
    meta = HistGradientBoostingClassifier(random_state=7).fit(panel(Xb), yb)

    print("== holdout ladder (11,371 hu / 20,000 ai; frozen models, ROC read) ==")
    ladder("exam ensemble", meta.predict_proba(panel(X_ai))[:, 1],
           meta.predict_proba(panel(X_hu))[:, 1])

    # --- tuned panel HGB (the head-to-head incumbent) on the full cache ---
    hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.08,
                                         max_features=0.5, random_state=7).fit(Xa, ya)
    ladder("panel HGB", hgb.predict_proba(X_ai)[:, 1], hgb.predict_proba(X_hu)[:, 1])

    # (L0 arm removed - offline per Donk 2026-08-17; resurrectable, see docstring)


if __name__ == "__main__":
    main()
