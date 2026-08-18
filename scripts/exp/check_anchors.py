"""Find reinsertion anchors for the two dropped columns."""
import numpy as np

names = list(np.load("data/derived/full_features.npz")["feature_names"])
for i, n in enumerate(names):
    if "para_len" in n or n.startswith("cv_") or n == "initial_char_entropy":
        print(i, n)
