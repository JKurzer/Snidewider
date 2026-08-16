"""Precompute base-detector scores on buckets A/B/C -> data/derived/base_scores.npz.

Fleet agents load this instead of recomputing burst/qgram/exemplar features
(~2 min each). Scores come from HGB models trained on bucket A; B and C are
pure inference. Usage: .venv\\Scripts\\python scripts/cache_base_scores.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import burst, qgram
from ai_text_detection.dct_shapes import dct_tail_vector
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import ExemplarBank, exemplar_vector
from ai_text_detection.features import qgram_profile_features
from ai_text_detection.features_relative import document_features_relative

N_BANK = 150


def relative_vector(text: str) -> list[float]:
    feats = document_features_relative(text)
    return [feats[k] for k in sorted(feats)]


def qgram12_vector(text: str) -> list[float]:
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
        short["mean"], short["stdev"], mid_ck2["mean"], mid_ck2["stdev"],
        mid_qg["mean"], mid_qg["stdev"], q2["qgram_entropy"], q3["qgram_entropy"],
        q3["qgram_distinct_ratio"], q3["qgram_repeat_frac"], q3["qgram_max_share"],
        top10_share,
    ]


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {name: (sub.model != "human").to_numpy(int) for name, sub in buckets.items()}

    a = buckets["A"]
    bank_ai = ExemplarBank.from_texts([str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts([str(t) for t in a[a.model == "human"].generation[:N_BANK]])

    fns = {
        "relative-burst": relative_vector,
        "qgram12": qgram12_vector,
        "exemplar": lambda t: exemplar_vector(qgram.profile(t.encode("utf-8"), 3), bank_ai, bank_hu),
        "dct-nobase": dct_tail_vector,
    }
    store = {f"labels_{b}": labels[b] for b in buckets}
    for det, fn in fns.items():
        Xa = np.array([fn(str(t)) for t in buckets["A"].generation], dtype=float)
        col_means = np.nanmean(Xa, axis=0)
        bad = np.where(~np.isfinite(Xa))
        Xa[bad] = np.take(col_means, bad[1])
        model = HistGradientBoostingClassifier(random_state=7).fit(Xa, labels["A"])
        for bucket in ("A", "B", "C"):
            X = np.array([fn(str(t)) for t in buckets[bucket].generation], dtype=float)
            bad = np.where(~np.isfinite(X))
            X[bad] = np.take(col_means, bad[1])
            store[f"{det}_{bucket}"] = model.predict_proba(X)[:, 1]
        print(f"{det} done")
    np.savez("data/derived/base_scores.npz", **store)
    print("data/derived/base_scores.npz written")


if __name__ == "__main__":
    main()
