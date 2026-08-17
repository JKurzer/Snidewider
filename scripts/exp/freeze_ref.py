"""One-shot: freeze the derived char reference into charstat.py."""
path = r"src\ai_text_detection\charstat.py"

# recompute the ref exactly as wire_distill_features.py does (full precision)
import string
from collections import Counter

import pandas as pd

from ai_text_detection.evaldata import split_buckets

df = pd.read_parquet("data/derived/raid_splits.parquet")
a_hu = split_buckets(df)["A"]
a_hu = a_hu[a_hu.model == "human"]
counts: Counter = Counter()
for t in a_hu.generation:
    counts.update(str(t).lower())
total = sum(counts.values())
ref = {c: counts.get(c, 0) / total for c in string.printable[:95]}
lit = "{" + ", ".join(f"{c!r}: {p!r}" for c, p in sorted(ref.items()) if p > 0) + "}"
src = open(path, encoding="utf-8").read()
import re
m = re.search(r"ENGLISH_CHAR_REF: dict\[str, float\] = \{[^\n]*\}", src)
assert m, "anchor missing"
old = m.group(0)
src = src.replace(old, "ENGLISH_CHAR_REF: dict[str, float] = " + lit)
open(path, "w", encoding="utf-8", newline="\n").write(src)

import importlib

import ai_text_detection.charstat as cs

importlib.reload(cs)
print("frozen:", len(cs.ENGLISH_CHAR_REF), "entries, sum =",
      round(sum(cs.ENGLISH_CHAR_REF.values()), 6))
