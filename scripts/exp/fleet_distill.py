"""FLEET H — single-pass distillation packs (collapse + charstat), benched.

No rolling comparisons anywhere: collapse = distribution-tail/vocabulary
stats per doc; charstat = character census + chi2 vs an A-human char
reference (dense low-dim statistic — in-fold risk negligible vs the n-gram
coverage case, but watch the increments).

Phase 1: solo AUROC/TPR@1e-2 per feature per bucket (rank on B).
Phase 2: HGB increment — panel(120) vs panel+distill(156), train A, read C.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_distill.py
"""

from __future__ import annotations

import string
from collections import Counter

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.charstat import CHARSTAT_FEATURE_NAMES, charstat_features
from ai_text_detection.collapse import COLLAPSE_FEATURE_NAMES, collapse_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_distill.md"
FEATS = [f"col_{n}" for n in COLLAPSE_FEATURE_NAMES] + \
        [f"chr_{n}" for n in CHARSTAT_FEATURE_NAMES]


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    a_hu = buckets["A"][buckets["A"].model == "human"]
    ref_counts: Counter = Counter()
    for t in a_hu.generation:
        ref_counts.update(str(t).lower())
    total = sum(ref_counts.values())
    ref = {c: ref_counts.get(c, 0) / total for c in string.printable[:95]}

    vals: dict[str, dict[str, np.ndarray]] = {}
    for b in "ABC":
        rows = []
        for t in buckets[b].generation:
            text = str(t)
            col = collapse_features(text)
            chr_ = charstat_features(text, ref)
            rows.append([col[k] for k in COLLAPSE_FEATURE_NAMES]
                        + [chr_[k] for k in CHARSTAT_FEATURE_NAMES])
        vals[b] = np.array(rows)
        print(f"bucket {b} done", flush=True)

    rows = []
    for j, f in enumerate(FEATS):
        row = {"feature": f}
        for b in "ABC":
            cov, roc, tpr = eval_feat(vals[b][:, j], labels[b])
            row[f"roc_{b}"], row[f"tpr_{b}"] = roc, tpr
        rows.append(row)
    rows.sort(key=lambda r: np.nan_to_num(r["roc_B"]), reverse=True)

    lines = ["# FLEET H — distillation packs (collapse + char census)\n\n",
             "| feature | AUROC A | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n",
             "|---|---|---|---|---|---|\n"]
    for r in rows:
        lines.append(f"| {r['feature']} | {r['roc_A']:.3f} | {r['roc_B']:.3f} | "
                     f"{r['roc_C']:.3f} | {r['tpr_B']:.3f} | {r['tpr_C']:.3f} |\n")

    # ---- phase 2: HGB increment ----
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

    lines.append("\n## HGB increment (train A, read B/C)\n\n")
    lines.append("| arm | n | AUROC B | TPR@1e-2 B | AUROC C | TPR@1e-2 C |\n|---|---|---|---|---|---|\n")
    for arm, get in (
        ("panel120", lambda b: prep(Xp[b])),
        ("panel+distill156", lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
        ("distill36 alone", lambda b: prep_new(b)),
    ):
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(get("A"), labels["A"])
        row = [arm, str(get("A").shape[1])]
        for b in ("B", "C"):
            s = m.predict_proba(get(b))[:, 1]
            roc = auroc(list(s[labels[b] == 1]), list(s[labels[b] == 0]))
            r = tpr_at_fpr(list(s[labels[b] == 1]), list(s[labels[b] == 0]), fpr=1e-2)
            row.extend([f"{roc:.3f}", f"{r['tpr']:.3f} [{r['tpr_lo']:.3f},{r['tpr_hi']:.3f}]"])
        lines.append("| " + " | ".join(row) + " |\n")
        print(lines[-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines[:40]))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
