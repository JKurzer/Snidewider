"""Micro-bench: construct_im cost per CSA type on a real 1.4KB doc."""
import pandas as pd

from ai_text_detection import _csa_native
from ai_text_detection.evaldata import split_buckets

df = pd.read_parquet("data/derived/raid_splits.parquet")
c = split_buckets(df)["C"]
t = str(c.generation.iloc[0]).encode("utf-8")
print("doc bytes:", len(t))
print(_csa_native.bench_construct(t, 100))
