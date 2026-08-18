"""Derive the top-K char trigrams from bucket-A humans (to freeze into chargrams.py)."""
from collections import Counter

import pandas as pd

from ai_text_detection.evaldata import split_buckets

K = 32

df = pd.read_parquet("data/derived/raid_splits.parquet")
a_hu = split_buckets(df)["A"]
a_hu = a_hu[a_hu.model == "human"]

counts: Counter = Counter()
for t in a_hu.generation:
    s = str(t).lower()
    counts.update(zip(s, s[1:], s[2:]))
top = counts.most_common(K)
for r, (tg, c) in enumerate(top, 1):
    print(f"{r:>3} {tg!r} {c}")
print("\nTRIGRAM_LIST = [" + ", ".join(repr("".join(tg)) for tg, _ in top) + "]")
