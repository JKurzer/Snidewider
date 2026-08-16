"""F1 experiment: doc-length-relative windows, full dev-fold coverage.

Baseline (features.py, absolute windows) dropped 4869/6000 dev docs (midrange
needs >= 350 tokens) and kept qgram_total, a length proxy confounded with the
label (RAID AI docs are shorter). This run: document_features_relative on ALL
6000 docs (target: zero dropped), no qgram_total.

Protocol: dev fold only; humans = all model=='human'; AI = sample 4000
(random_state=17); 50/50 split by source_id (RandomState(23)); logreg +
HistGB; AUROC + TPR@FPR=1e-3 with Wilson CI.

Run:  set PYTHONPATH=..\\src && <venv>\\python scripts\\exp_f1.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ai_text_detection
from ai_text_detection.features_relative import (
    FEATURE_NAMES_RELATIVE,
    document_features_relative,
)
from ai_text_detection.metrics import auroc, tpr_at_fpr

PARQUET = r"C:\Users\poly\ai-text-detection\data\derived\raid_splits.parquet"
N_AI = 4000


def load_dev_docs() -> pd.DataFrame:
    df = pd.read_parquet(
        PARQUET, columns=["source_id", "model", "generation", "fold"]
    )
    dev = df[df.fold == "dev"]  # holdout fold untouched (RULES: never tune on it)
    humans = dev[dev.model == "human"]
    ais = dev[dev.model != "human"].sample(n=N_AI, random_state=17)
    docs = pd.concat([humans, ais]).reset_index(drop=True)
    print(f"dev docs: humans={len(humans)} ai={len(ais)} total={len(docs)}")
    return docs


def featurize(docs: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rows = []
    for i, text in enumerate(docs.generation):
        rows.append(document_features_relative(text))
        if (i + 1) % 1000 == 0:
            print(f"  featurized {i + 1}/{len(docs)} ({time.time() - t0:.1f}s)")
    X = pd.DataFrame(rows, columns=list(FEATURE_NAMES_RELATIVE))
    print(f"featurization: {time.time() - t0:.1f}s for {len(X)} docs")
    return X


def split_by_source(docs: pd.DataFrame) -> np.ndarray:
    """50/50 split by source_id (never by chunk — RULES #4). Returns is_test."""
    sources = pd.unique(docs.source_id)
    perm = np.random.RandomState(23).permutation(len(sources))
    test_sources = set(sources[perm[: len(sources) // 2]])
    return docs.source_id.isin(test_sources).to_numpy()


def evaluate(name: str, scores: np.ndarray, y_test: np.ndarray) -> dict:
    ai = scores[y_test == 1].tolist()
    hu = scores[y_test == 0].tolist()
    auc = auroc(ai, hu)
    t = tpr_at_fpr(ai, hu, 1e-3)
    print(
        f"{name:>6}: AUROC={auc:.4f}  TPR@1e-3={t['tpr']:.3f} "
        f"[{t['tpr_lo']:.3f}, {t['tpr_hi']:.3f}]  (n_ai={len(ai)}, n_hu={len(hu)})"
    )
    return {"model": name, "auroc": auc, "tpr": t["tpr"], "lo": t["tpr_lo"], "hi": t["tpr_hi"]}


def per_feature_auroc(X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    print("\nper-feature AUROC (test; '*' = inverted, lower = more AI-like):")
    rows = []
    for col in X_test.columns:
        vals = X_test[col].to_numpy()
        auc = auroc(vals[y_test == 1].tolist(), vals[y_test == 0].tolist())
        rows.append((col, auc))
    for col, auc in sorted(rows, key=lambda r: -max(r[1], 1 - r[1])):
        flip = "*" if auc < 0.5 else " "
        print(f"  {col:>22}: {max(auc, 1 - auc):.4f}{flip}")


def main() -> None:
    print("ai_text_detection from:", ai_text_detection.__file__)
    assert "f1-relative-windows" in ai_text_detection.__file__, "wrong worktree!"

    docs = load_dev_docs()
    X = featurize(docs)
    n_nan = int(X.isna().any(axis=1).sum())
    print(f"docs with any NaN feature: {n_nan} (dropped: 0)")
    if n_nan:
        X = X.fillna(X.median())  # degenerate micro-docs only; count reported above

    y = (docs.model != "human").to_numpy()
    is_test = split_by_source(docs)
    X_tr, X_te = X[~is_test], X[is_test]
    y_tr, y_te = y[~is_test], y[is_test]
    print(f"split: train={len(X_tr)} (ai={y_tr.sum()}) test={len(X_te)} (ai={y_te.sum()})")

    results = []
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    logreg.fit(X_tr, y_tr)
    results.append(evaluate("logreg", logreg.predict_proba(X_te)[:, 1], y_te))

    hgb = HistGradientBoostingClassifier(random_state=0)
    hgb.fit(X_tr, y_tr)
    results.append(evaluate("hgb", hgb.predict_proba(X_te)[:, 1], y_te))

    per_feature_auroc(X_te, y_te)

    print("\nlogreg coefficients (standardized, sorted by |coef|):")
    coefs = logreg.named_steps["logisticregression"].coef_[0]
    for name, c in sorted(zip(X.columns, coefs), key=lambda r: -abs(r[1])):
        print(f"  {name:>22}: {c:+.4f}")

    print("\n== headline (full-coverage, n=6000, zero dropped) ==")
    for r in results:
        print(
            f"  {r['model']:>6}: AUROC={r['auroc']:.4f} "
            f"TPR@1e-3={r['tpr']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
        )
    print("baseline (biased 1131-doc subset): logreg AUROC=0.898 TPR=0.202 | "
          "hgb AUROC=0.925 TPR=0.261")


if __name__ == "__main__":
    main()
