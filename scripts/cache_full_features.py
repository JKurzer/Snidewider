"""Cache full per-doc feature matrices (all detector families) for A/B/C.

Stores data/derived/full_features.npz: feature names, labels, and the raw
81-feature matrix per bucket. Regenerate when feature definitions change.
Usage: .venv\\Scripts\\python scripts/cache_full_features.py
"""

import numpy as np
import pandas as pd

from ai_text_detection import qgram
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import EXEMPLAR_FEATURE_NAMES, ExemplarBank, exemplar_vector
from ai_text_detection.feature_sets import QGRAM12_NAMES, qgram12_vector, relative_vector
from ai_text_detection.features_relative import FEATURE_NAMES_RELATIVE

N_BANK = 150


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    a = buckets["A"]
    bank_ai = ExemplarBank.from_texts([str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts([str(t) for t in a[a.model == "human"].generation[:N_BANK]])

    names = (
        [f"rel_{n}" for n in sorted(FEATURE_NAMES_RELATIVE)]
        + [f"qg_{n}" for n in QGRAM12_NAMES]
        + list(EXEMPLAR_FEATURE_NAMES)
        + [f"dct_{n}" for n in sorted(dct_tail_features("sample text. another one.").keys())]
    )
    store: dict[str, np.ndarray] = {}
    for bucket, sub in buckets.items():
        rows = []
        for text in sub.generation:
            text = str(text)
            rel = relative_vector(text)
            qg = qgram12_vector(text)
            ex = exemplar_vector(qgram.profile(text.encode("utf-8"), 3), bank_ai, bank_hu)
            tail = dct_tail_features(text)
            rows.append(rel + qg + ex + [tail[k] for k in sorted(tail)])
        store[f"X_{bucket}"] = np.array(rows, dtype=float)
        store[f"y_{bucket}"] = (sub.model != "human").to_numpy(int)
        print(f"{bucket}: {store[f'X_{bucket}'].shape}")
    store["feature_names"] = np.array(names)
    np.savez("data/derived/full_features.npz", **store)
    print(f"data/derived/full_features.npz written ({len(names)} features)")


if __name__ == "__main__":
    main()
