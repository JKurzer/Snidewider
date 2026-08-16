"""D5 probe: how stable is the 4-det hgb-meta TPR@1e-3?

The fleet recorded 0.155 for 4det raw / hgb; exp_d5 measures 0.234. Check
whether the tail is just meta-learner noise: permute meta-feature column
order (fleet put dct 3rd, exp_d5 puts it 4th), vary hgb random_state, and
look at the DCT score tail on B (the miscalibration hypothesis itself).

Run like exp_d5.py (same PYTHONPATH/cwd).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp_d5 import BASE_DETS, dct_matrix, report

from ai_text_detection.evaldata import split_buckets


def hgb_stack(zb, zc, labels_b, seed):
    meta = HistGradientBoostingClassifier(random_state=seed).fit(zb, labels_b)
    return meta.predict_proba(zc)[:, 1]


def main():
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    store = np.load("data/derived/base_scores.npz")
    labels = {n: (s.model != "human").to_numpy(int) for n, s in buckets.items()}

    feats, col_means = {}, None
    for name in ("A", "B", "C"):
        feats[name], col_means = dct_matrix(buckets[name].generation, col_means)
    model = HistGradientBoostingClassifier(random_state=7).fit(feats["A"], labels["A"])
    raw_b = {det: store[f"{det}_B"] for det in BASE_DETS} | {"dct": model.predict_proba(feats["B"])[:, 1]}
    raw_c = {det: store[f"{det}_C"] for det in BASE_DETS} | {"dct": model.predict_proba(feats["C"])[:, 1]}
    yb, yc = labels["B"], labels["C"]

    # --- the tail hypothesis itself: where do DCT's top humans sit? (bucket B)
    hu, ai = raw_b["dct"][yb == 0], raw_b["dct"][yb == 1]
    print("dct B-score quantiles (raw hgb prob):")
    print(f"  human  p50 {np.median(hu):.3f}  p99 {np.quantile(hu, 0.99):.3f}  max {hu.max():.3f}")
    print(f"  ai     p50 {np.median(ai):.3f}  p10 {np.quantile(ai, 0.10):.3f}  p01 {np.quantile(ai, 0.01):.3f}")
    top = np.quantile(hu, 0.999)
    print(f"  AI share above human p99.9 ({top:.3f}): {(ai >= top).mean():.3f}")

    # --- column-order sensitivity (fleet order vs exp_d5 order), raw scores
    orders = {
        "fleet order (dct 3rd)": ("relative-burst", "qgram12", "dct", "exemplar"),
        "exp_d5 order (dct 4th)": ("relative-burst", "qgram12", "exemplar", "dct"),
    }
    print("\n== column-order sensitivity: 4det raw / hgb(seed=7) ==")
    for tag, order in orders.items():
        zb = np.column_stack([raw_b[d] for d in order])
        zc = np.column_stack([raw_c[d] for d in order])
        report(f"{tag}", yc, hgb_stack(zb, zc, yb, 7))

    # --- seed sensitivity on the exp_d5 column order
    print("\n== seed sensitivity: 4det raw / hgb, dct 4th ==")
    for seed in (0, 1, 2, 7, 42):
        order = orders["exp_d5 order (dct 4th)"]
        zb = np.column_stack([raw_b[d] for d in order])
        zc = np.column_stack([raw_c[d] for d in order])
        report(f"hgb seed={seed}", yc, hgb_stack(zb, zc, yb, seed))

    # --- logreg invariance check (should be identical across orders)
    print("\n== logreg on both orders (linear => order-invariant) ==")
    for tag, order in orders.items():
        zb = np.column_stack([raw_b[d] for d in order])
        zc = np.column_stack([raw_c[d] for d in order])
        meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(zb, yb)
        report(f"logreg {tag}", yc, meta.predict_proba(zc)[:, 1])


if __name__ == "__main__":
    main()
