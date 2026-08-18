"""FLEET X — covering number + within-doc density: solo (both tails) + increment."""
import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.cover import COVER_FEATURE_NAMES, cover_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_cover.md"


def tail3(vals, y):
    m = np.isfinite(vals)
    if m.sum() < 100:
        return float("nan")
    pos, neg = vals[m & (y == 1)], vals[m & (y == 0)]
    r = tpr_at_fpr(list(pos), list(neg), fpr=1e-3)
    r2 = tpr_at_fpr(list(-pos), list(-neg), fpr=1e-3)
    return max(r["tpr"], r2["tpr"])


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals = {}
    for b in "ABC":
        vals[b] = np.array([[cover_features(str(t))[k] for k in COVER_FEATURE_NAMES]
                            for t in buckets[b].generation])
        print(f"{b} done", flush=True)

    lines = ["# FLEET X — covering number + within-doc density (solo)\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C | TPR@1e-3 B | TPR@1e-3 C |\n|---|---|---|---|---|---|---|\n"]
    for j, f in enumerate(COVER_FEATURE_NAMES):
        rb = eval_feat(vals["B"][:, j], labels["B"])
        rc = eval_feat(vals["C"][:, j], labels["C"])
        t3b = tail3(vals["B"][:, j], labels["B"])
        t3c = tail3(vals["C"][:, j], labels["C"])
        lines.append(f"| {f} | {rb[1]:.3f} | {rc[1]:.3f} | {rb[2]:.3f} | {rc[2]:.3f} | "
                     f"{t3b:.3f} | {t3c:.3f} |\n")

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
                     (f"panel+{n_panel + 3}",
                      lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("cover3 alone", lambda b: prep_new(b))):
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
