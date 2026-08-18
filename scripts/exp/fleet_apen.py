"""FLEET U — Approximate/Sample Entropy of the byte stream (Pincus 1991,
Richman & Moorman 2000). Regularity of local fluctuation: low ApEn =
predictable patterning. Byte-level only (ApEn needs a symbol metric;
token ids have none). m=2, r=0.2*std, Chebyshev via cKDTree.

Solo bench + increment over the 250 panel.
"""
import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_apen.md"
FEATS = ("apen_char", "sampen_char")
M = 2


def _block(series: np.ndarray, m: int) -> np.ndarray:
    return np.column_stack([series[i:len(series) - m + 1 + i] for i in range(m)])


def apen_features(text: str) -> dict[str, float]:
    b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)
    n = len(b)
    if n < 400:
        return {k: np.nan for k in FEATS}
    r = 0.2 * float(b.std())
    if r == 0:
        return {k: np.nan for k in FEATS}

    counts = []
    pair_counts = []
    for m in (M, M + 1):
        X = _block(b, m)
        tree = cKDTree(X)
        balls = tree.query_ball_point(X, r, p=np.inf, workers=-1)
        c = np.array([len(ball) for ball in balls], dtype=float)
        counts.append(c / len(X))
        # SampEn pair counts (i<j, no self matches)
        pair_counts.append(sum(len(ball) - 1 for ball in balls) / 2)

    phi = [np.log(c).mean() for c in counts]
    apen = float(phi[0] - phi[1])
    b_pairs, a_pairs = pair_counts
    sampen = float(-np.log(a_pairs / b_pairs)) if a_pairs > 0 and b_pairs > 0 else np.nan
    return {"apen_char": apen, "sampen_char": sampen}


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals = {}
    for b in "ABC":
        vals[b] = np.array([[apen_features(str(t))[k] for k in FEATS]
                            for t in buckets[b].generation])
        print(f"{b} done", flush=True)

    lines = ["# FLEET U — ApEn/SampEn of the byte stream (solo)\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n|---|---|---|---|---|\n"]
    for j, f in enumerate(FEATS):
        rb = eval_feat(vals["B"][:, j], labels["B"])
        rc = eval_feat(vals["C"][:, j], labels["C"])
        lines.append(f"| {f} | {rb[1]:.3f} | {rc[1]:.3f} | {rb[2]:.3f} | {rc[2]:.3f} |\n")

    panel = np.load("data/derived/full_features.npz")
    n_panel = len(panel["feature_names"])
    Xp = {b: panel[f"X_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(Xp["A"], axis=0))

    def prep(X):
        X = X.copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(means, bad[1])
        return X

    new_means = np.nan_to_num(np.nanmean(vals["A"], axis=0))

    def prep_new(b):
        X = vals[b].copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(new_means, bad[1])
        return X

    lines.append("\n## HGB increment (train A, read C)\n\n")
    lines.append("| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |\n|---|---|---|---|---|\n")
    for arm, get in ((f"panel{n_panel}", lambda b: prep(Xp[b])),
                     (f"panel+{n_panel + 2}",
                      lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("apen2 alone", lambda b: prep_new(b))):
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(get("A"), labels["A"])
        s = m.predict_proba(get("C"))[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        r1 = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]), fpr=1e-2)
        r3 = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]), fpr=1e-3)
        lines.append(f"| {arm} | {get('A').shape[1]} | {roc:.4f} | "
                     f"{r1['tpr']:.3f} [{r1['tpr_lo']:.3f},{r1['tpr_hi']:.3f}] | "
                     f"{r3['tpr']:.3f} |\n")
        print(lines[-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
