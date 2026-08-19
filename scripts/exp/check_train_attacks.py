"""Enumerate the RAID train split's attack/setting structure."""
from urllib.request import urlopen

r = urlopen("https://dataset.raid-bench.xyz/train.csv")
print(f"train.csv: {int(r.headers['Content-Length'])/1e6:.1f} MB")
r.close()

import pandas as pd

from raid.utils import load_data

train = load_data(split="train", include_adversarial=True)
print(f"rows: {len(train)}")
print("columns:", list(train.columns))
for col in ("attack", "decoding", "repetition_penalty", "split"):
    if col in train.columns:
        print(f"\n{col}:")
        print(train[col].value_counts().to_string())
