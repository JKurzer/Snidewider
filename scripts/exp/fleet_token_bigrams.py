"""FLEET O — token bigrams: top-64 rates + peak-reuse pack, bench + increment."""
import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr
from ai_text_detection.token_bigrams import (
    REUSE_FEATURE_NAMES,
    TOKEN_BIGRAM_FEATURE_NAMES,
    token_bigram_rates,
    token_reuse_features,
)

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_token_bigrams.md"
FEATS = list(TOKEN_BIGRAM_FEATURE_NAMES) + list(REUSE_FEATURE_NAMES)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals = {}
    for b in "ABC":
        rows = []
        for t in buckets[b].generation:
            text = str(t)
            r = token_bigram_rates(text)
            ru = token_reuse_features(text)
            rows.append([r[k] for k in TOKEN_BIGRAM_FEATURE_NAMES]
                        + [ru[k] for k in REUSE_FEATURE_NAMES])
        vals[b] = np.array(rows)
        print(f"{b} done", flush=True)

    rows_out = []
    for j, f in enumerate(FEATS):
        rb = eval_feat(vals["B"][:, j], labels["B"])
        rc = eval_feat(vals["C"][:, j], labels["C"])
        rows_out.append((f, rb, rc))
    rows_out.sort(key=lambda r: np.nan_to_num(r[1][1]), reverse=True)

    lines = ["# FLEET O — token bigrams (rates + peak reuse); top 15 by B AUROC\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n|---|---|---|---|---|\n"]
    for f, rb, rc in rows_out[:15]:
        lines.append(f"| {f} | {rb[1]:.3f} | {rc[1]:.3f} | {rb[2]:.3f} | {rc[2]:.3f} |\n")
    lines.append("\n## reuse pack (all)\n\n| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n|---|---|---|---|---|\n")
    for f, rb, rc in rows_out:
        if f in REUSE_FEATURE_NAMES:
            lines.append(f"| {f} | {rb[1]:.3f} | {rc[1]:.3f} | {rb[2]:.3f} | {rc[2]:.3f} |\n")

    panel = np.load("data/derived/full_features.npz")
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
    lines.append("| arm | n | AUROC C | TPR@1e-2 C |\n|---|---|---|---|\n")
    for arm, get in (("panel221", lambda b: prep(Xp[b])),
                     ("panel+tg290", lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("tg69 alone", lambda b: prep_new(b))):
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(get("A"), labels["A"])
        s = m.predict_proba(get("C"))[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        r = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]), fpr=1e-2)
        lines.append(f"| {arm} | {get('A').shape[1]} | {roc:.3f} | "
                     f"{r['tpr']:.3f} [{r['tpr_lo']:.3f},{r['tpr_hi']:.3f}] |\n")
        print(lines[-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
