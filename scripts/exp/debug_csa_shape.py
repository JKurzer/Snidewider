"""Repro: run the CSA-block profile over the first N C-bucket docs with progress."""
import sys

import pandas as pd

sys.path.insert(0, "scripts/exp")
from fleet_csa_shape import profile

from ai_text_detection.evaldata import split_buckets

df = pd.read_parquet("data/derived/raid_splits.parquet")
c = split_buckets(df)["C"]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
for i, t in enumerate(c.generation[:limit]):
    b = str(t).encode("utf-8")
    p = profile(b)
    if i % 20 == 0:
        print(f"{i}: len={len(b)} blocks={len(p)}", flush=True)
print("SURVIVED", limit)
