"""Show the DCT inputs (token-length series) for a human doc and an AI doc."""

import re

import pandas as pd

from ai_text_detection.shape import dct_run_map

df = pd.read_parquet("data/derived/raid_splits.parquet")
dev = df[(df.fold == "dev") & (df.domain == "news")]
human = dev[dev.model == "human"].iloc[0]
ai = dev[dev.model != "human"].iloc[0]

for label, text in (("HUMAN", human.generation), ("AI", ai.generation)):
    tokens = re.findall(r"[A-Za-z0-9']+", str(text))
    lengths = [min(len(w), 15) for w in tokens]
    print(f"===== {label} =====")
    print("first 300 chars:", str(text)[:300].replace("\n", " "))
    print()
    print("first 48 token lengths:", lengths[:48])
    print("dct_run_map symbols (first 12):", list(dct_run_map(str(text))[:12]))
    print()
