"""F1 ablation: why did the baseline look good — subset bias or qgram_total?

Arms (same 6000 dev docs, same RandomState(23) source split, same models):
  A absolute+drop    — features.py as-is, NaN docs dropped (baseline protocol)
  B absolute+impute  — full coverage, NaN -> train median, keeps qgram_total
  C relative+qtotal  — relative windows + qgram_total restored
  D relative         — exp_f1 headline arm (reprinted for convenience)

A vs B isolates the biased-subset effect; B/C vs D isolates qgram_total;
B vs C isolates the cost of relativizing the windows.

Run AFTER exp_f1.py. Imports its loaders (same directory, no package tricks).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exp_f1 import evaluate, load_dev_docs, split_by_source

from ai_text_detection.features import FEATURE_NAMES, document_features
from ai_text_detection.features_relative import (
    FEATURE_NAMES_RELATIVE,
    document_features_relative,
)


def featurize_both(docs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    abs_rows, rel_rows = [], []
    for text in docs.generation:
        abs_rows.append(document_features(text))
        rel_rows.append(document_features_relative(text))
    print(f"featurization (both sets): {time.time() - t0:.1f}s")
    return (
        pd.DataFrame(abs_rows, columns=list(FEATURE_NAMES)),
        pd.DataFrame(rel_rows, columns=list(FEATURE_NAMES_RELATIVE)),
    )


def run_arm(name: str, X: pd.DataFrame, y: np.ndarray, is_test: np.ndarray) -> None:
    X_tr, X_te = X[~is_test], X[is_test]
    y_tr, y_te = y[~is_test], y[is_test]
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    logreg.fit(X_tr, y_tr)
    hgb = HistGradientBoostingClassifier(random_state=0)
    hgb.fit(X_tr, y_tr)
    print(f"--- arm {name} (train={len(X_tr)}, test={len(X_te)}) ---")
    evaluate("logreg", logreg.predict_proba(X_te)[:, 1], y_te)
    evaluate("hgb", hgb.predict_proba(X_te)[:, 1], y_te)


def main() -> None:
    docs = load_dev_docs()
    X_abs, X_rel = featurize_both(docs)
    y = (docs.model != "human").to_numpy()

    # Arm A: baseline protocol — drop NaN docs first, then split the remainder.
    ok = ~X_abs.isna().any(axis=1)
    print(f"arm A: {int((~ok).sum())}/{len(docs)} docs dropped (NaN windows)")
    run_arm(
        "A absolute+drop",
        X_abs[ok].reset_index(drop=True),
        y[ok.to_numpy()],
        split_by_source(docs[ok].reset_index(drop=True)),
    )

    # Arms B/C/D: full coverage, single shared split.
    is_test = split_by_source(docs)
    med = X_abs[~is_test].median()
    run_arm("B absolute+impute", X_abs.fillna(med), y, is_test)
    X_rel_filled = X_rel.fillna(X_rel[~is_test].median())
    X_c = X_rel_filled.copy()
    X_c["qgram_total"] = X_abs["qgram_total"]
    run_arm("C relative+qtotal", X_c, y, is_test)
    run_arm("D relative", X_rel_filled, y, is_test)

    # standalone discriminative power of the dropped length proxy, full coverage
    from ai_text_detection.metrics import auroc

    qt = X_abs["qgram_total"].to_numpy()
    print(
        "qgram_total alone, full test set: AUROC="
        f"{auroc(qt[is_test & (y == 1)].tolist(), qt[is_test & (y == 0)].tolist()):.4f}"
    )


if __name__ == "__main__":
    main()
