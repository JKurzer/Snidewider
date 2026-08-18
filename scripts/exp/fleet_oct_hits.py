"""FLEET R3 — oct_hits (Donk's spec) increment over the 226 base and 268 panel."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr
from ai_text_detection.token_bigrams import OCT_HITS_FEATURE_NAMES, oct_hits_features

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_oct_hits.md"


def main() -> None:
    d = np.load("data/derived/full_features.npz")
    names = list(d["feature_names"])
    is_pack = [n.startswith(("tg3_", "cv_", "bwt_")) or n == "initial_char_entropy"
               for n in names]
    base_idx = [i for i, p in enumerate(is_pack) if not p]

    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    oct_vals = {}
    for b in "ABC":
        oct_vals[b] = np.array([[oct_hits_features(str(t))[k] for k in OCT_HITS_FEATURE_NAMES]
                                for t in buckets[b].generation])
    oct_means = np.nan_to_num(np.nanmean(oct_vals["A"], axis=0))

    X = {b: d[f"X_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(X["A"], axis=0))
    for b in "ABC":
        bad = np.where(~np.isfinite(X[b]))
        X[b][bad] = np.take(means, bad[1])
        ob = oct_vals[b].copy()
        bad = np.where(~np.isfinite(ob))
        ob[bad] = np.take(oct_means, bad[1])
        oct_vals[b] = ob

    arms = {
        "226 base": lambda b: X[b][:, base_idx],
        "228 base+oct": lambda b: np.column_stack([X[b][:, base_idx], oct_vals[b]]),
        "268 current": lambda b: X[b],
        "270 current+oct": lambda b: np.column_stack([X[b], oct_vals[b]]),
    }

    lines = ["# FLEET R3 — oct_hits increment (train A, read C)\n\n",
             "| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |\n|---|---|---|---|---|\n"]
    for arm, get in arms.items():
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
