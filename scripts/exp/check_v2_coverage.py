"""v2 (window=64, min_gap=16) coverage check: what fraction of docs get a
full 256-sample series now?"""
import numpy as np
import pandas as pd

from ai_text_detection.burst import random_change_series
from ai_text_detection.evaldata import split_buckets

c = split_buckets(pd.read_parquet("data/derived/raid_splits.parquet"))["C"]
texts = [str(t) for t in c.generation[:400]]

full = partial = empty = 0
for t in texts:
    s = random_change_series(t, window=64, samples=256, min_gap=16,
                             metric="ck2", unit="tokens")
    if not s:
        empty += 1
    elif len(s) < 256:
        partial += 1
    else:
        full += 1
n = len(texts)
print(f"v2 (64/16): full {full/n:.1%} | partial {partial/n:.1%} | empty {empty/n:.1%}")
print(f"(v1 (150/50) was 24.5% full, 75.5% empty)")
