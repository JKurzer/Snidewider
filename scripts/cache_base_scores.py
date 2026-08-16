"""Precompute base-detector scores on buckets A/B/C -> data/derived/base_scores.npz.

Fleet agents load this instead of recomputing burst/qgram/exemplar features
(~2 min each). Scores come from HGB models trained on bucket A; B and C are
pure inference. Usage: .venv\\Scripts\\python scripts/cache_base_scores.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import qgram
from ai_text_detection.dct_shapes import dct_tail_vector
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import ExemplarBank, bank_self_indices, exemplar_vector
from ai_text_detection.feature_sets import qgram12_vector, relative_vector

N_BANK = 150


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
        "dct-nobase": dct_tail_vector,
    }
    # leave-one-out for bucket A rows that sit inside the exemplar banks
    ai_self, hu_self = bank_self_indices([str(m) for m in buckets["A"].model], N_BANK)

    def rows_for(det: str, bucket: str) -> np.ndarray:
        sub = buckets[bucket]
        if det == "exemplar":
            if bucket == "A":
                return np.array([exemplar_vector(qgram.profile(str(t).encode("utf-8"), 3),
                                                 bank_ai, bank_hu, ai_self[i], hu_self[i])
                                 for i, t in enumerate(sub.generation)], dtype=float)
            return np.array([exemplar_vector(qgram.profile(str(t).encode("utf-8"), 3),
                                             bank_ai, bank_hu)
                             for t in sub.generation], dtype=float)
        return np.array([fns[det](str(t)) for t in sub.generation], dtype=float)

    store = {f"labels_{b}": labels[b] for b in buckets}
    for det in (*fns, "exemplar"):
        Xa = rows_for(det, "A")
        col_means = np.nanmean(Xa, axis=0)
        bad = np.where(~np.isfinite(Xa))
        Xa[bad] = np.take(col_means, bad[1])
        model = HistGradientBoostingClassifier(random_state=7).fit(Xa, labels["A"])
        for bucket in ("A", "B", "C"):
            X = rows_for(det, bucket)
            bad = np.where(~np.isfinite(X))
            X[bad] = np.take(col_means, bad[1])
            store[f"{det}_{bucket}"] = model.predict_proba(X)[:, 1]
        print(f"{det} done")
    np.savez("data/derived/base_scores.npz", **store)
    print("data/derived/base_scores.npz written")


if __name__ == "__main__":
    main()
