"""Per-block timing of the post-rebuild additions (ablation, one at a time)."""
import time

import numpy as np
import pandas as pd

from ai_text_detection import _csa_native, pipeline, qgram
from ai_text_detection.bwt_stats import bwt_features
from ai_text_detection.chargrams import chargram_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.token_bigrams import oct_hits_features

c = split_buckets(pd.read_parquet("data/derived/raid_splits.parquet"))["C"]
texts = [str(t) for t in c.generation[:40]]

# warmup
for t in texts[:10]:
    _csa_native.csa_stats(t.encode("utf-8"))
    pipeline._delta_row(t)
    oct_hits_features(t)
    chargram_features(t)


def timed(fn, label):
    t0 = time.perf_counter()
    for t in texts:
        fn(t)
    print(f"  {label:<28} {(time.perf_counter() - t0) / len(texts) * 1000:6.2f} ms/doc")


print("per-block (40 C docs, warm):")
timed(lambda t: _csa_native.csa_stats(t.encode("utf-8")), "csa_stats (bwt's engine)")
timed(lambda t: pipeline._delta_row(t), "_delta_row (8 qgram profiles)")
timed(lambda t: oct_hits_features(t), "oct_hits")
timed(lambda t: chargram_features(t), "chargram (cv + initial)")
timed(lambda t: [qgram.profile(t.encode('utf-8'), k) for k in range(1, 9)],
      "8x qgram.profile raw")
