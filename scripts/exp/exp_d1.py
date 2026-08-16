"""D1 width sweep: which DCT width K makes the stack better at the tail?

Grid: K in {1,2,3,4,6,8,12,16} x unit in {sentences, windows(24)}.
Question: does any K push the 4-detector HGB stack past TPR@FPR=1e-3 = 0.170
(the 3-detector reference) without giving up AUROC (0.896)?

Protocol (leakage-proof, mirrors scripts/stack_detectors.py):
  DCT detector = HistGradientBoostingClassifier(random_state=7) trained on
    bucket A's 4 DCT features; NaNs left as NaN (HGB handles them natively —
    imputing from B/C stats would be hidden state, RULES #5).
  Stack = HGB(random_state=7) trained on bucket B's 4 scores (cached
    relative-burst / qgram12 / exemplar + this config's DCT score).
  All final numbers read once on bucket C. Holdout fold untouched.

Fast path: one tokenize+embed pass per doc per unit, dct.dct_coefficients per
(segment, K). Verified feature-for-feature against dct.dct_features on a
sample before the sweep runs (RULES #2: no eval, no ship).

Usage (cwd = main checkout, data paths are relative):
  set PYTHONPATH=..\\src&& ..\\.venv\\Scripts\\python scripts\\exp_d1.py
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import ai_text_detection
from ai_text_detection import dct
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

KS = (1, 2, 3, 4, 6, 8, 12, 16)
UNITS = ("sentences", "windows")
WINDOW = 24
N_VERIFY = 150  # docs cross-checked against dct.dct_features per config

CACHED = ("relative-burst", "qgram12", "exemplar")


def segments_for(text: str, unit: str) -> list[str]:
    """Mirror dct.dct_features' segmentation exactly (single source: dct.py)."""
    if unit == "sentences":
        return [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 2]
    tokens = text.split()
    return [
        " ".join(tokens[i : i + WINDOW])
        for i in range(0, len(tokens) - WINDOW + 1, WINDOW)
    ]


def featurize(texts: list[str], unit: str) -> dict[int, np.ndarray]:
    """(n_docs, 4) feature matrix per K, one embed pass per doc.

    Replicates dct.dct_features semantics: skip segments with <2 embedded
    tokens, NaN row when fewer than two usable segments, order-energy ratio
    ||c[1]||/||c[0]|| (0.0 for K=1). Verified against dct.dct_features below.
    """
    feats = {k: np.full((len(texts), len(dct.DCT_FEATURE_NAMES)), np.nan) for k in KS}
    for row, text in enumerate(texts):
        coeffs_per_segment = []
        for segment in segments_for(text, unit):
            embedded = dct.embed_sentence(segment)
            if embedded.shape[0] < 2:
                continue
            coeffs_per_segment.append(dct.dct_coefficients(embedded, max(KS)))
        if len(coeffs_per_segment) < 2:
            continue
        for k in KS:
            vectors = [c[:k].reshape(-1) for c in coeffs_per_segment]
            adjacent = [
                dct._cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
            ]
            energies = []
            for c in coeffs_per_segment:
                base = float(np.linalg.norm(c[0]))
                energies.append(float(np.linalg.norm(c[1]) / base) if base and k > 1 else 0.0)
            feats[k][row] = (
                float(np.mean(adjacent)),
                float(np.std(adjacent)),
                float(np.mean(energies)),
                float(np.std(energies)),
            )
    return feats


def verify_fast_path(texts: list[str]) -> None:
    """Fast path must reproduce dct.dct_features for every (K, unit) config."""
    sampled = texts[:N_VERIFY]
    fast = {unit: featurize(sampled, unit) for unit in UNITS}
    n_checked = 0
    for unit in UNITS:
        for k in KS:
            for row, text in enumerate(sampled):
                ref = dct.dct_features(text, k=k, unit=unit, window=WINDOW)
                for col, name in enumerate(dct.DCT_FEATURE_NAMES):
                    got, want = fast[unit][k][row, col], ref[name]
                    if math.isnan(want):
                        assert math.isnan(got), f"{unit=} {k=} {row=} {name}: expected NaN"
                    else:
                        assert math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-7), (
                            f"{unit=} {k=} {row=} {name}: {got} != {want}"
                        )
                    n_checked += 1
    print(f"fast path verified vs dct.dct_features: {n_checked} values match")


def report(tag: str, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    ai, human = list(scores[labels == 1]), list(scores[labels == 0])
    roc = auroc(ai, human)
    res = tpr_at_fpr(ai, human)
    print(
        f"  {tag:<28} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} "
        f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]"
    )
    return {"auroc": roc, "tpr": res["tpr"], "lo": res["tpr_lo"], "hi": res["tpr_hi"]}


def main() -> None:
    print("import from:", ai_text_detection.__file__)
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    cached = np.load("data/derived/base_scores.npz")
    labels = {}
    for name, sub in buckets.items():
        bucket_labels = (sub.model != "human").to_numpy(int)
        assert np.array_equal(bucket_labels, cached[f"labels_{name}"]), (
            f"bucket {name}: row order disagrees with base_scores.npz"
        )
        labels[name] = cached[f"labels_{name}"]
        print(f"bucket {name}: {len(sub)} docs ({int(labels[name].sum())} ai)")

    texts = {name: [str(t) for t in sub.generation] for name, sub in buckets.items()}
    verify_fast_path(texts["A"] + texts["B"][: N_VERIFY // 2])

    print("\nfeaturizing (2 units x 3 buckets, one embed pass each)...")
    features = {}
    for unit in UNITS:
        for name in buckets:
            features[(unit, name)] = featurize(texts[name], unit)
            print(f"  {unit} / bucket {name} done")

    print("\n== 3-detector reference stack (cached scores only, HGB meta on B, read on C) ==")
    z_b = np.column_stack([cached[f"{det}_B"] for det in CACHED])
    z_c = np.column_stack([cached[f"{det}_C"] for det in CACHED])
    meta = HistGradientBoostingClassifier(random_state=7)
    meta.fit(z_b, labels["B"])
    reference = report("reference-3det", meta.predict_proba(z_c)[:, 1], labels["C"])

    print("\n== sweep: DCT solo on C, then 4-detector stack on C ==")
    rows = []
    for unit in UNITS:
        for k in KS:
            x_a = features[(unit, "A")][k]
            detector = HistGradientBoostingClassifier(random_state=7)
            detector.fit(x_a, labels["A"])
            dct_scores = {
                name: detector.predict_proba(features[(unit, name)][k])[:, 1]
                for name in ("B", "C")
            }
            tag = f"K={k:<2} {unit}"
            solo = report(f"{tag} solo", dct_scores["C"], labels["C"])
            meta = HistGradientBoostingClassifier(random_state=7)
            meta.fit(np.column_stack([z_b, dct_scores["B"]]), labels["B"])
            stacked = report(
                f"{tag} stacked",
                meta.predict_proba(np.column_stack([z_c, dct_scores["C"]]))[:, 1],
                labels["C"],
            )
            rows.append({"k": k, "unit": unit, "solo": solo, "stacked": stacked})

    print("\n== summary (reference: AUROC {auroc:.3f} TPR {tpr:.3f}) ==".format(**reference))
    print(f"{'K':>3} {'unit':<10} {'solo AUROC':>10} {'solo TPR':>9} "
          f"{'stack AUROC':>11} {'stack TPR':>10}")
    for r in rows:
        print(f"{r['k']:>3} {r['unit']:<10} {r['solo']['auroc']:>10.3f} "
              f"{r['solo']['tpr']:>9.3f} {r['stacked']['auroc']:>11.3f} "
              f"{r['stacked']['tpr']:>10.3f}")
    best = max(rows, key=lambda r: (r["stacked"]["tpr"], r["stacked"]["auroc"]))
    clears = best["stacked"]["tpr"] > reference["tpr"]
    keeps_roc = best["stacked"]["auroc"] >= reference["auroc"] - 1e-3
    print(f"\nbest tail: K={best['k']} {best['unit']} - stacked TPR {best['stacked']['tpr']:.3f} "
          f"(ref {reference['tpr']:.3f}), AUROC {best['stacked']['auroc']:.3f} "
          f"(ref {reference['auroc']:.3f})")
    print(f"clears TPR 0.170: {clears} | keeps AUROC: {keeps_roc}")


if __name__ == "__main__":
    main()
