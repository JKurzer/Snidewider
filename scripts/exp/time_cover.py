"""Execution-time investigation, proper protocol: warmup + 5 alternating
rounds, median per mode (PERF-RULES: measure, never guess)."""
import time

import numpy as np
import pandas as pd

from ai_text_detection import pipeline
from ai_text_detection.cover import cover_features
from ai_text_detection.evaldata import split_buckets

c = split_buckets(pd.read_parquet("data/derived/raid_splits.parquet"))["C"]
texts = [str(t) for t in c.generation[:60]]
art = pipeline.load_artifacts()

# warmup (native registries, allocator arenas)
for t in texts[:20]:
    pipeline.featurize(t, art)
    pipeline.featurize(t, art, csa_mode="full")
    cover_features(t)


def timed(fn, n=60):
    t0 = time.perf_counter()
    for t in texts[:n]:
        fn(t)
    return (time.perf_counter() - t0) / n * 1000


rounds = {"impute": [], "full": [], "cover": []}
for r in range(5):
    rounds["impute"].append(timed(lambda t: pipeline.featurize(t, art)))
    rounds["full"].append(timed(lambda t: pipeline.featurize(t, art, csa_mode="full")))
    rounds["cover"].append(timed(cover_features, n=30))

for k, v in rounds.items():
    v = np.array(v)
    print(f"{k:>7}: median {np.median(v):6.2f} ms/doc | rounds "
          + " ".join(f"{x:.1f}" for x in v))
