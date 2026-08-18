"""FLEET S — the delta rematch: the 13-feature delta family vs the modern panel.

Historical verdict (fleet K, 153-era panel): condensate increment TIED and the
pack was benched (RULE 6). Donk: 'we were probably hasty; tastier options at
the time.' Rematch against the rebuilt panel (train A, read C).
"""
import numpy as np
import pandas as pd
from fleet_condensates import DELTA_NAMES, delta_feats
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_delta_rematch.md"


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    dvals = {}
    for b in "ABC":
        dvals[b] = np.array([[delta_feats(str(t))[k] for k in DELTA_NAMES]
                             for t in buckets[b].generation])
        print(f"{b} deltas done", flush=True)

    panel = np.load("data/derived/full_features.npz")
    n_panel = len(panel["feature_names"])
    Xp = {b: panel[f"X_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(Xp["A"], axis=0))

    def prep(X):
        X = X.copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(means, bad[1])
        return X

    new_means = np.nan_to_num(np.nanmean(dvals["A"], axis=0))

    def prep_new(b):
        X = dvals[b].copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(new_means, bad[1])
        return X

    lines = ["# FLEET S — delta rematch vs the modern panel (train A, read C)\n\n",
             "| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |\n|---|---|---|---|---|\n"]
    for arm, get in ((f"panel{n_panel}", lambda b: prep(Xp[b])),
                     (f"panel+{n_panel + 13}",
                      lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("delta13 alone", lambda b: prep_new(b))):
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
