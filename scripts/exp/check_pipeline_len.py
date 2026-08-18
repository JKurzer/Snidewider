"""Find the featurize-vs-cache column delta by direct measurement."""
import numpy as np
import pandas as pd

from ai_text_detection import pipeline
from ai_text_detection.evaldata import split_buckets

art = pipeline.load_artifacts()
cache_names = list(np.load("data/derived/full_features.npz")["feature_names"])
print(f"cache: {len(cache_names)}")

c = split_buckets(pd.read_parquet("data/derived/raid_splits.parquet"))["C"]
row = pipeline.featurize(str(c.generation.iloc[0]), art, csa_mode="full")
print(f"featurize: {len(row)}")

# reconstruct what featurize THINKS the names are, block by block
feat_set = set(art["feature_names"])
from ai_text_detection.chargrams import CHARGRAM_FEATURE_NAMES
from ai_text_detection.bwt_stats import BWT_FEATURE_NAMES
ex_names = [k for k in art["feature_names"]
            if k.startswith("ex_") and k != "ex_contrast_centroid"]
print(f"ex block: {len(ex_names)}")
print(f"chargram emitted: {[k for k in CHARGRAM_FEATURE_NAMES if k in feat_set]}")
print(f"bwt emitted: {BWT_FEATURE_NAMES}")
print("appended singles:",
      [n for n in ("qg_s256_ck2_mean", "oct_hits", "ex_contrast_centroid",
                   "ex_contrast_mode") if n in feat_set])
