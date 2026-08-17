"""Derive the top-K word bigrams from bucket-A humans (to freeze into token_bigrams.py)."""
from collections import Counter

import pandas as pd

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.stats_features import WORD_RE

K = 64

df = pd.read_parquet("data/derived/raid_splits.parquet")
a_hu = split_buckets(df)["A"]
a_hu = a_hu[a_hu.model == "human"]

counts: Counter = Counter()
for t in a_hu.generation:
    toks = [w.lower() for w in WORD_RE.findall(str(t))]
    counts.update(zip(toks, toks[1:]))
top = counts.most_common(K)
print(f"top {K} token bigrams:")
for r, (bg, c) in enumerate(top, 1):
    print(f"{r:>3} {bg} {c}")
print("\nTOKEN_BIGRAM_LIST = [" + ", ".join(repr(bg) for bg, _ in top) + "]")
