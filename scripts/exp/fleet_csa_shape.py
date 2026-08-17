"""FLEET L — CSA compressibility PROFILE shape features.

Donk's call: the signal is not recoverable by the mean. So: per-doc block
compressibility series (csa_wt rate per ~1/8 block), summarized by SHAPE,
not level:
  shape_stdev / shape_iqr / shape_range   spread of local compressibility
  shape_arc                               first-half minus last-half (drift of rate)
  shape_maxmin                            worst block / best block ratio
  shape_cv                                stdev / |mean|

Humans write lumpy text (bursty compressibility); machines write smooth.
Solo bench + increment over the 156 panel. DEV ONLY.
Usage: .venv\\Scripts\\python scripts\\exp\\fleet_csa_shape.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import _csa_native
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_csa_shape.md"
FEATS = ["shape_stdev", "shape_iqr", "shape_range", "shape_arc",
         "shape_maxmin", "shape_cv"]


def profile(b: bytes, nblocks: int = 8) -> np.ndarray:
    n = len(b)
    if n < 800:
        return np.array([])
    step = n // nblocks
    rates = []
    for j in range(nblocks):
        blk = b[j * step:(j + 1) * step if j < nblocks - 1 else n]
        rates.append(_csa_native.csa_stats(blk)["csa_wt_bytes"] / len(blk))
    return np.array(rates)


def shape_feats(text: str) -> dict[str, float]:
    p = profile(text.encode("utf-8"))
    if len(p) < 4:
        return {f: np.nan for f in FEATS}
    q1, q3 = np.percentile(p, [25, 75])
    half = len(p) // 2
    return {
        "shape_stdev": float(p.std()),
        "shape_iqr": float(q3 - q1),
        "shape_range": float(p.max() - p.min()),
        "shape_arc": float(p[:half].mean() - p[-half:].mean()),
        "shape_maxmin": float(p.max() / max(1e-9, p.min())),
        "shape_cv": float(p.std() / max(1e-9, abs(p.mean()))),
    }


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals = {}
    for b in "ABC":
        rows = []
        for i, t in enumerate(buckets[b].generation):
            rows.append([shape_feats(str(t))[f] for f in FEATS])
            if i % 250 == 0:
                with open(r"scripts\exp\_csa_shape_progress.log", "a") as fh:
                    fh.write(f"{b} {i}\n")
        vals[b] = np.array(rows)
        with open(r"scripts\exp\_csa_shape_progress.log", "a") as fh:
            fh.write(f"{b} done\n")
        print(f"{b} done", flush=True)

    lines = ["# FLEET L — CSA compressibility-profile shape features\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n|---|---|---|---|---|\n"]
    for j, f in enumerate(FEATS):
        rb = eval_feat(vals["B"][:, j], labels["B"])
        rc = eval_feat(vals["C"][:, j], labels["C"])
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
    for arm, get in (("panel156", lambda b: prep(Xp[b])),
                     ("panel+shape162", lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("shape6 alone", lambda b: prep_new(b))):
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
