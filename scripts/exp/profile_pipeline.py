"""PROFILE: per-family cost of the 157-feat pipeline (measure, never guess)."""
import time

import numpy as np
import pandas as pd

from ai_text_detection import _csa_native, burst, pipeline, qgram
from ai_text_detection.charstat import charstat_features
from ai_text_detection.collapse import collapse_features
from ai_text_detection.coverage import coverage_features
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import exemplar_vector
from ai_text_detection.feature_sets import qgram12_vector, relative_vector
from ai_text_detection.shape import shape_features
from ai_text_detection.stats_features import stat_features

N = 60  # docs


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    c = split_buckets(df)["C"]
    art = pipeline.load_artifacts()
    texts = [str(t) for t in c.generation[:N]]

    fams = {
        "relative": lambda t: relative_vector(t),
        "qgram12": lambda t: qgram12_vector(t),
        "exemplar": lambda t: exemplar_vector(qgram.profile(t.encode(), 3),
                                              art["bank_ai"], art["bank_hu"]),
        "dct_tail": lambda t: dct_tail_features(t),
        "shape": lambda t: shape_features(t),
        "stats": lambda t: stat_features(t),
        "coverage": lambda t: coverage_features(t, art["ref_hu"], art["ref_ai"]),
        "collapse": lambda t: collapse_features(t),
        "charstat": lambda t: charstat_features(t),
        "csa": lambda t: _csa_native.csa_stats(t.encode()),
        "s256": lambda t: burst.random_change_series(t, window=150, samples=256,
                                                     min_gap=50, metric="ck2",
                                                     unit="tokens"),
    }

    print(f"per-family cost, median ms/doc over {N} C-bucket docs\n")
    totals = {}
    for name, fn in fams.items():
        times = []
        for t in texts:
            t0 = time.perf_counter()
            fn(t)
            times.append((time.perf_counter() - t0) * 1000)
        med = float(np.median(times))
        totals[name] = med
        print(f"  {name:<10} {med:8.1f} ms   (p90 {np.percentile(times, 90):.1f})")

    t0 = time.perf_counter()
    for t in texts[:20]:
        pipeline.featurize(t, art)
    full = (time.perf_counter() - t0) / 20 * 1000
    print(f"\nfull pipeline: {full:.1f} ms/doc | sum of family medians: "
          f"{sum(totals.values()):.1f} ms")
    print("\nslowest first:", sorted(totals.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
