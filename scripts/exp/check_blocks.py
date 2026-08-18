"""Count the current cache's family blocks and check the surgery DROP names."""
import numpy as np

names = list(np.load("data/derived/full_features.npz")["feature_names"])
print(f"total: {len(names)}")
for n in ("ex_contrast_mean", "shape_skeleton_step_mean", "dct_arc_cos"):
    print(f"  dropped? {n}: {'STILL PRESENT' if n in names else 'gone'}")
for pref in ("rel_", "qg_", "ex_", "dct_", "shape_", "stat_", "cov", "col_",
             "chr_", "csa_", "bg_", "reuse_", "tg3_", "cv_", "bwt_", "oct_"):
    c = [n for n in names if n.startswith(pref)]
    if pref == "ex_":
        c = [n for n in c if n not in ("ex_contrast_mode", "ex_contrast_centroid")]
    if pref == "qg_":
        c = [n for n in c if not n.startswith("qg_s256")]
    if c:
        print(f"  {pref}: {len(c)}")
print("  qg_s256:", sum(1 for n in names if n.startswith("qg_s256")))
print("  initial:", sum(1 for n in names if n == "initial_char_entropy"))
print("  centroid:", sum(1 for n in names if n == "ex_contrast_centroid"))
