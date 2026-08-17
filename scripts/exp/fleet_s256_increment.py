"""Increment trial: does qg_w150_s256_ck2_mean earn a panel seat?

The fleet-A2 champion holds the holdout deep-tail record solo (0.200@1e-3)
but was never tested against the 156 panel. Arms: panel vs panel+the stat,
HGB tuned config, train A, read C (dev only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import burst
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)


def s256_mean(text: str) -> float:
    s = burst.random_change_series(text, window=150, samples=256, min_gap=50,
                                   metric="ck2", unit="tokens")
    return float(np.mean(s)) if s else np.nan


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    panel = np.load("data/derived/full_features.npz")
    y = {b: panel[f"y_{b}"] for b in "ABC"}
    X = {b: panel[f"X_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(X["A"], axis=0))
    for b in "ABC":
        bad = np.where(~np.isfinite(X[b]))
        X[b][bad] = np.take(means, bad[1])
        new = np.array([s256_mean(str(t)) for t in buckets[b].generation])
        nm = np.nan_to_num(np.nanmean(new[np.isfinite(new)])) if np.isfinite(new).any() else 0.0
        new[~np.isfinite(new)] = nm
        X[b] = np.column_stack([X[b], new])
        print(f"{b} stat computed", flush=True)

    for arm, cols in (("panel156", slice(0, 156)), ("panel+ s256", slice(None))):
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X["A"][:, cols], y["A"])
        s = m.predict_proba(X["C"][:, cols])[:, 1]
        roc = auroc(list(s[y["C"] == 1]), list(s[y["C"] == 0]))
        for fpr in (1e-2, 1e-3):
            r = tpr_at_fpr(list(s[y["C"] == 1]), list(s[y["C"] == 0]), fpr=fpr)
            print(f"{arm:<12} C AUROC {roc:.4f} TPR@{fpr:.0e} {r['tpr']:.3f} "
                  f"[{r['tpr_lo']:.3f},{r['tpr_hi']:.3f}]", flush=True)


if __name__ == "__main__":
    main()
