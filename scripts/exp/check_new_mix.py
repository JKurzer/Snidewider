"""Sanity: the stratified dev AI mix should look like holdout's uniform mix."""
import pandas as pd

from ai_text_detection.evaldata import split_buckets

df = pd.read_parquet("data/derived/raid_splits.parquet")
b = split_buckets(df)
ai = pd.concat([b[x] for x in "ABC"])
ai = ai[ai.model != "human"]
print("new dev AI model mix:")
print(ai.model.value_counts(normalize=True).round(3).to_string())
print("rows/source mean:", round(ai.groupby("source_id").size().mean(), 3))
print("bucket sizes:", {x: len(b[x]) for x in "ABC"})
