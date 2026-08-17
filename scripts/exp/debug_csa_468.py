"""Reproduce the suspected poison doc (C bucket row 468, len 1448)."""
import sys

import pandas as pd

from ai_text_detection import _csa_native
from ai_text_detection.evaldata import split_buckets

df = pd.read_parquet("data/derived/raid_splits.parquet")
c = split_buckets(df)["C"]
b = str(c.generation.iloc[468]).encode("utf-8")
print("doc len:", len(b), flush=True)
print("doc head:", b[:120], flush=True)

print("whole-doc csa_stats...", flush=True)
out = _csa_native.csa_stats(b)
print("whole ok:", out["n"], flush=True)

n = len(b)
step = n // 8
for j in range(8):
    blk = b[j * step:(j + 1) * step if j < 7 else n]
    print(f"block {j}: len={len(blk)} head={blk[:30]!r}", flush=True)
    out = _csa_native.csa_stats(blk)
    print(f"block {j} ok", flush=True)
print("ALL BLOCKS SURVIVED", flush=True)
