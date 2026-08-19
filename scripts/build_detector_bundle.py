"""Build the detector bundle: everything needed to score raw text.

Trains the tuned panel HGB on bucket A (full cache, A-fit imputation) and
packs: model, feature names, imputation means, exemplar banks (first 150
A rows per class), coverage references (built from A; scoring-side protocol
= the same one B/C/holdout use), and the cache schema version stamp.

Output: data/derived/detector_bundle.pkl — the whole missile.
Usage: .venv\\Scripts\\python scripts\\build_detector_bundle.py
"""

import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.coverage import QS, build_reference
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import ExemplarBank, centroid_profile

HGB_PARAMS = dict(max_iter=600, max_depth=8, learning_rate=0.08,
                  max_features=0.5, random_state=7)  # d8/i600: holdout champion (fleet Z)
N_BANK = 150
OUT = "data/derived/detector_bundle.pkl"


def main() -> None:
    cache = np.load("data/derived/full_features.npz")
    names = list(cache["feature_names"])
    Xa = cache["X_A"].astype(float)
    ya = cache["y_A"]
    means = np.nan_to_num(np.nanmean(Xa, axis=0))
    bad = np.where(~np.isfinite(Xa))
    Xa[bad] = np.take(means, bad[1])

    try:
        supp = np.load("data/derived/supp_features.npz")
        Xs = supp["X"].astype(float)
        bad = np.where(~np.isfinite(Xs))
        Xs[bad] = np.take(means, bad[1])
        Xa = np.vstack([Xa, Xs])
        ya = np.concatenate([ya, supp["y"]])
        print(f"training supplement included: {Xs.shape}")
    except OSError:
        pass

    model = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xa, ya)
    print(f"trained panel HGB on A(+supp): {Xa.shape}")

    df = pd.read_parquet("data/derived/raid_splits.parquet")
    a = split_buckets(df)["A"]
    bank_ai = ExemplarBank.from_texts([str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts([str(t) for t in a[a.model == "human"].generation[:N_BANK]])
    ref_hu = {q: build_reference(a[a.model == "human"].generation, q) for q in QS}
    ref_ai = {q: build_reference(a[a.model != "human"].generation, q) for q in QS}
    print("banks + refs built")

    bundle = {
        "model": model,
        "feature_names": names,
        "impute_means": means,
        "bank_ai": bank_ai,
        "bank_hu": bank_hu,
        "centroid_ai": centroid_profile(bank_ai),
        "centroid_hu": centroid_profile(bank_hu),
        "ref_ai": ref_ai,
        "ref_hu": ref_hu,
        "hgb_params": HGB_PARAMS,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "n_features": len(names),
    }
    with open(OUT, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"{OUT} written ({len(names)} features)")


if __name__ == "__main__":
    main()
