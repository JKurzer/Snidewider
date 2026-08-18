"""Verify the restore anchors exist in the current cache names."""
import numpy as np

names = list(np.load("data/derived/full_features.npz")["feature_names"])
print([n for n in names if "paircos" in n])
print([n for n in names if n.startswith("cov")])
