"""Head-to-head: L0/L0L2 best-subset logistic (_l0learn_native) vs the HGB forest.

Same cache, same buckets, same A-fit imputation, same metrics as the incumbent
(train_forest.py). Protocol (RULES #3/#4): all fits on bucket A; the L0 path
point (penalty, gamma row, lambda column) is selected on bucket B ONLY, by
TPR@FPR with AUROC as tie-break; bucket C is read exactly once, for the
champion and the incumbent. Holdout cache stays locked.

FPR=1e-2, not 1e-3: at ~750 humans/bucket, 1e-3 allows k=0 false positives
(the metric degenerates to the zero-FP gate stat); 1e-2 gives k=7 and real
tail resolution.

Usage: .venv\\Scripts\\python scripts\\exp\\exp_l0_vs_hgb.py
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import _l0learn_native
from ai_text_detection.metrics import auroc, tpr_at_fpr, zero_fpr_tpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
N_LAMBDA = 100
MAX_NNZ = 25
FPR = 1e-2


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def report(tag: str, scores: np.ndarray, y: np.ndarray) -> None:
    ai, hu = list(scores[y == 1]), list(scores[y == 0])
    roc = auroc(ai, hu)
    res = tpr_at_fpr(ai, hu, fpr=FPR)
    gate = zero_fpr_tpr(ai, hu)
    print(f"  {tag:<22} AUROC {roc:.3f} | TPR@{FPR:.0e} {res['tpr']:.3f} "
          f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}] (FPRa {res['fpr_achieved']:.1e}) "
          f"| zero-FP TPR {gate['tpr']:.3f}")


def main() -> None:
    data = np.load("data/derived/full_features.npz")
    names = list(data["feature_names"])
    X = {b: data[f"X_{b}"].astype(np.float64) for b in "ABC"}
    y = {b: data[f"y_{b}"].astype(np.float64) for b in "ABC"}

    col_means = np.nan_to_num(np.nanmean(X["A"], axis=0))  # A-fit, applied to all
    for b in "ABC":
        bad = np.where(~np.isfinite(X[b]))
        X[b][bad] = np.take(col_means, bad[1])

    # --- incumbent: the HGB forest, hyperparams frozen in train_forest.py ---
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"], y["A"])
    print("== incumbent HGB ==")
    report("B (select)", hgb.predict_proba(X["B"])[:, 1], y["B"])

    # --- challenger: best-subset logistic paths on A ---
    candidates = []  # (penalty, nnz, beta, b0)
    for penalty in ("L0", "L0L2"):
        out = _l0learn_native.fit(np.asfortranarray(X["A"]), y["A"], penalty=penalty,
                                  n_lambda=N_LAMBDA, max_nnz=MAX_NNZ)
        offset = 0
        for Bmat in out["betas"]:
            Bmat = np.asarray(Bmat)
            plen = Bmat.shape[1]
            for j in range(plen):
                beta = Bmat[:, j]
                candidates.append((penalty, int(np.count_nonzero(beta)), beta,
                                   float(out["intercepts"][offset + j])))
            offset += plen
    print(f"\n== challenger: {len(candidates)} path points (L0 + L0L2), selecting on B ==")

    scored = []
    for penalty, nnz, beta, b0 in candidates:
        sb = sigmoid(X["B"] @ beta + b0)
        res = tpr_at_fpr(list(sb[y["B"] == 1]), list(sb[y["B"] == 0]), fpr=FPR)
        roc = auroc(list(sb[y["B"] == 1]), list(sb[y["B"] == 0]))
        scored.append(((res["tpr"], roc), penalty, nnz, beta, b0, res["tpr_lo"], res["tpr_hi"]))
    scored.sort(key=lambda t: t[0], reverse=True)

    print("top 5 on B (selection landscape):")
    for (tpr_b, roc_b), penalty, nnz, _, _, lo, hi in scored[:5]:
        print(f"  {penalty:<4} nnz={nnz:<3} TPR@{FPR:.0e} {tpr_b:.3f} [{lo:.3f}, {hi:.3f}] AUROC {roc_b:.3f}")

    (_, roc_b), penalty, nnz, beta, b0, _, _ = scored[0]
    picked = [names[i] for i in np.nonzero(beta)[0]]
    print(f"\nchampion: {penalty} nnz={nnz} (B AUROC {roc_b:.3f}) -> {picked}")

    print("\n== FINAL read on C (once) ==")
    report("HGB", hgb.predict_proba(X["C"])[:, 1], y["C"])
    report(f"L0 champ ({penalty},{nnz})", sigmoid(X["C"] @ beta + b0), y["C"])


if __name__ == "__main__":
    main()
