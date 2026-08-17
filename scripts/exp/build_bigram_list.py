"""Derive the top-K char bigrams from bucket-A humans (to freeze into bigrams.py)."""
import string
from collections import Counter

import pandas as pd

from ai_text_detection.evaldata import split_buckets

K = 64

df = pd.read_parquet("data/derived/raid_splits.parquet")
a_hu = split_buckets(df)["A"]
a_hu = a_hu[a_hu.model == "human"]

counts: Counter = Counter()
for t in a_hu.generation:
    s = str(t).lower()
    counts.update(zip(s, s[1:]))
top = counts.most_common(K)
print(f"top {K} bigrams (rank, bigram, count):")
for r, (bg, c) in enumerate(top, 1):
    print(f"{r:>3} {bg!r} {c}")
print("\nBIGRAM_LIST = [" + ", ".join(repr(a + b) for (a, b), _ in top) + "]")
