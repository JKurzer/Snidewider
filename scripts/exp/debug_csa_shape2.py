"""Poison-pill hunter: per-doc progress to a FILE; last line before death wins."""
import sys

import pandas as pd

sys.path.insert(0, "scripts/exp")
from fleet_csa_shape import profile

from ai_text_detection.evaldata import split_buckets

LOG = r"scripts\exp\_csa_shape_progress.log"

df = pd.read_parquet("data/derived/raid_splits.parquet")
c = split_buckets(df)["C"]
with open(LOG, "w", encoding="utf-8") as fh:
    for i, t in enumerate(c.generation):
        b = str(t).encode("utf-8")
        fh.write(f"{i} {len(b)}\n")
        fh.flush()
        try:
            p = profile(b)
        except Exception as exc:
            fh.write(f"EXC {i} {type(exc).__name__} {exc}\n")
            fh.flush()
    fh.write("SURVIVED ALL\n")
print("done")
