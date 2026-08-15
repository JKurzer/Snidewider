"""Stacked classifiers over separated detectors.

Detectors (each its own feature family + own model, never blended):
  relative-burst — F1's length-relative window features (full coverage)
  qgram12        — F3's recommended 12-feature set
  exemplar       — F4's doc-vs-corpus contrast features

Protocol (leakage-proof): dev sources split A/B/C (50/25/25). Detectors
train on A, score B; meta-classifiers train on B's detector scores; final
numbers are on C, once. Exemplar banks come from A only.
Usage: .venv\\Scripts\\python scripts/stack_detectors.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_text_detection import burst, dct, qgram
from ai_text_detection.evaldata import AI_PER_SOURCE, split_buckets
from ai_text_detection.exemplar import ExemplarBank, exemplar_vector
from ai_text_detection.features import qgram_profile_features
from ai_text_detection.features_relative import document_features_relative
from ai_text_detection.metrics import auroc, tpr_at_fpr

AI_PER_SOURCE = 2  # cap AI rows per source; every source contributes its human row
N_BANK = 150  # exemplars per class, from bucket A


def qgram12_vector(text: str) -> list[float]:
    """F3's recommended 12: ck2 short/mid stats, qgram-metric mid stats,
    q2/q3 entropies, q3 ratios, q5 top10 share."""
    short = burst.burst_features(text, window=64, gap=0, unit="bytes")
    mid_ck2 = burst.burst_features(
        text, window=150, samples=32, min_gap=50, unit="tokens", mode="random"
    )
    mid_qg = burst.burst_features(
        text, window=150, samples=32, min_gap=50, unit="tokens", mode="random", metric="qgram"
    )
    q2 = qgram_profile_features(text, 2)
    q3 = qgram_profile_features(text, 3)
    prof5 = qgram.profile(text.encode("utf-8"), 5)
    total5 = sum(c for _, c in prof5)
    top10_share = sum(c for _, c in prof5[:10]) / total5 if total5 else float("nan")
    return [
        short["mean"],
        short["stdev"],
        mid_ck2["mean"],
        mid_ck2["stdev"],
        mid_qg["mean"],
        mid_qg["stdev"],
        q2["qgram_entropy"],
        q3["qgram_entropy"],
        q3["qgram_distinct_ratio"],
        q3["qgram_repeat_frac"],
        q3["qgram_max_share"],
        top10_share,
    ]


def relative_vector(text: str) -> list[float]:
    feats = document_features_relative(text)
    return [feats[k] for k in sorted(feats)]  # stable order


def dct_vector(text: str) -> list[float]:
    feats = dct.dct_features(text)
    return [feats[name] for name in dct.DCT_FEATURE_NAMES]


DETECTORS = {
    "relative-burst": relative_vector,
    "qgram12": qgram12_vector,
    "dct": dct_vector,
}


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    for name, sub in buckets.items():
        print(f"bucket {name}: {len(sub)} docs ({(sub.model == 'human').sum()} human)")

    # --- featurize: burst detectors + exemplar profiles in one pass ---
    print("featurizing (relative + qgram12 + exemplar banks)...")
    feats = {name: {det: [] for det in DETECTORS} | {"exemplar": []} for name in buckets}
    labels, in_bank = {}, {}
    for name, sub in buckets.items():
        labels[name] = (sub.model != "human").to_numpy(int)
        in_bank[name] = np.zeros(len(sub), bool)
    bank_ai, bank_hu = None, None
    a = buckets["A"]
    a_ai_texts = [str(t) for t in a[a.model != "human"].generation[:N_BANK]]
    a_hu_texts = [str(t) for t in a[a.model == "human"].generation[:N_BANK]]
    bank_ai = ExemplarBank.from_texts(a_ai_texts)
    bank_hu = ExemplarBank.from_texts(a_hu_texts)
    print(f"exemplar banks: {len(bank_ai)} ai, {len(bank_hu)} human (from A)")

    for name, sub in buckets.items():
        for text in sub.generation:
            text = str(text)
            for det, fn in DETECTORS.items():
                feats[name][det].append(fn(text))
            feats[name]["exemplar"].append(
                exemplar_vector(qgram.profile(text.encode("utf-8"), 3), bank_ai, bank_hu)
            )
        print(f"  bucket {name} done")

    def matrix(bucket: str, detector: str) -> np.ndarray:
        X = np.array(feats[bucket][detector], dtype=float)
        col_means = np.nanmean(X, axis=0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])
        return X

    # --- train detectors on A, meta on B's scores, final numbers on C ---
    meta_rows = {}
    for det in feats["A"]:
        model = HistGradientBoostingClassifier(random_state=7)
        model.fit(matrix("A", det), labels["A"])
        meta_rows[det] = model
        for bucket in ("B", "C"):
            pass  # scores computed below
    scores = {}
    for det, model in meta_rows.items():
        scores[det] = {
            bucket: model.predict_proba(matrix(bucket, det))[:, 1] for bucket in ("B", "C")
        }

    print("\n== single-detector performance on C ==")
    for det in scores:
        s = scores[det]["C"]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        res = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        print(f"  {det:<15} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")

    Zb = np.column_stack([scores[d]["B"] for d in scores])
    Zc = np.column_stack([scores[d]["C"] for d in scores])
    print("\n== stacked meta-classifiers (trained on B, tested on C) ==")
    metas = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(random_state=7),
        "mean-vote": None,
    }
    for name, model in metas.items():
        if model is None:
            s = Zc.mean(axis=1)
        else:
            model.fit(Zb, labels["B"])
            s = model.predict_proba(Zc)[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        res = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        print(f"  {name:<15} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")
    if metas["logreg"] is not None:
        coefs = metas["logreg"].named_steps["logisticregression"].coef_[0]
        for det, c in zip(scores, coefs):
            print(f"  logreg weight {det:<15} {c:+.3f}")


if __name__ == "__main__":
    main()
