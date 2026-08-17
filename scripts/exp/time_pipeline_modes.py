"""Time impute vs full csa modes of the production pipeline."""
import time

import pandas as pd

from ai_text_detection import pipeline
from ai_text_detection.evaldata import split_buckets

art = pipeline.load_artifacts()
c = split_buckets(pd.read_parquet("data/derived/raid_splits.parquet"))["C"]
texts = [str(t) for t in c.generation[:60]]

t0 = time.perf_counter()
for t in texts:
    pipeline.featurize(t, art)
cheap = (time.perf_counter() - t0) / len(texts) * 1000

t0 = time.perf_counter()
for t in texts:
    pipeline.featurize(t, art, csa_mode="full")
full = (time.perf_counter() - t0) / len(texts) * 1000

print(f"impute mode (production): {cheap:.1f} ms/doc")
print(f"full mode (forensics):    {full:.1f} ms/doc")
