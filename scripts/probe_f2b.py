"""Probe 2: >=350-token distribution (the featurizable subset) + timing."""

import time

import pandas as pd

from ai_text_detection.features import feature_vector

df = pd.read_parquet(r"C:\Users\poly\ai-text-detection\data\derived\raid_splits.parquet")
dev = df[df.fold == "dev"]
humans = dev[dev.model == "human"]
ai = dev[dev.model != "human"].sample(n=4000, random_state=17)

tok_h = humans.generation.astype(str).str.split().str.len()
tok_a = ai.generation.astype(str).str.split().str.len()

bins = [350, 375, 400, 450, 500, 600, 800, 10**9]
labels = ["350-375", "375-400", "400-450", "450-500", "500-600", "600-800", "800+"]
tab = pd.DataFrame(
    {
        "human": pd.cut(tok_h, bins, labels=labels, right=False).value_counts().sort_index(),
        "ai": pd.cut(tok_a, bins, labels=labels, right=False).value_counts().sort_index(),
    }
)
print(tab.to_string())
print(f"total >=350: humans={int((tok_h >= 350).sum())} ai={int((tok_a >= 350).sum())}")

# timing on a handful of long docs
sample = pd.concat([humans.head(15), ai[ai.generation.astype(str).str.split().str.len() >= 350].head(15)])
t0 = time.perf_counter()
for t in sample.generation:
    feature_vector(str(t))
dt = time.perf_counter() - t0
print(f"\nfeaturized {len(sample)} docs in {dt:.1f}s -> {dt / len(sample) * 1000:.0f} ms/doc")
print(f"projected for 6000 docs: {dt / len(sample) * 6000 / 60:.1f} min")
