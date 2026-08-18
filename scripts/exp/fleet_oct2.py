"""FLEET R2 — the trigram question, measured four ways (cache now at 268).

Arms (train A, read C):
  226 base      : cache minus the chargram block (tg3_/cv_/initial_/bwt_)
  236 no-tri    : base + cv/initial/bwt (Donk's steer: trigrams out)
  238 donk      : no-tri + oct_repeat pair (his adjacent-octgram score)
  268 current   : the wired pack as it stands (trigrams in)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr
from ai_text_detection.token_bigrams import OCT_FEATURE_NAMES, oct_repeat_features

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_oct2.md"


def main() -> None:
    d = np.load("data/derived/full_features.npz")
    names = list(d["feature_names"])
    is_tri = [n.startswith("tg3_") for n in names]
    is_pack = [n.startswith(("tg3_", "cv_", "bwt_")) or n == "initial_char_entropy"
               for n in names]
    base_idx = [i for i, p in enumerate(is_pack) if not p]
    notri_idx = [i for i, (p, t) in enumerate(zip(is_pack, is_tri)) if not t]
    full_idx = list(range(len(names)))

    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}
    oct_vals = {}
    for b in "ABC":
        oct_vals[b] = np.array([[oct_repeat_features(str(t))[k] for k in OCT_FEATURE_NAMES]
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
        "236 no-tri": lambda b: X[b][:, notri_idx],
        "238 donk": lambda b: np.column_stack([X[b][:, notri_idx], oct_vals[b]]),
        "268 current": lambda b: X[b][:, full_idx],
    }

    lines = ["# FLEET R2 — trigram question, four arms (train A, read C)\n\n",
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
