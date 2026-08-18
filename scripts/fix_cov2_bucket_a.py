"""Fix the restore's cov2_ai on bucket A: recompute vs B+C refs (cross-bucket).

The restore mistakenly scored A rows against the bundle's A-built refs
(self-coverage -> saturated 1.0). Original convention (wire_stats_features):
A rows score vs B+C refs; B/C/holdout vs A refs (those are already correct).
"""

import numpy as np
import pandas as pd

from ai_text_detection.coverage import QS, build_reference, coverage_features
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    bc = pd.concat([buckets["B"], buckets["C"]])
    ref_hu = {q: build_reference(bc[bc.model == "human"].generation, q) for q in QS}
    ref_ai = {q: build_reference(bc[bc.model != "human"].generation, q) for q in QS}
    print("B+C refs built", flush=True)

    vals = np.array([coverage_features(str(t), ref_hu, ref_ai)["cov2_ai"]
                     for t in buckets["A"].generation], dtype=float)
    print(f"A cov2_ai recomputed (med {np.median(vals):.4f})", flush=True)

    cache = np.load(DEV_NPZ)
    names = list(cache["feature_names"])
    store = {k: cache[k] for k in cache}
    store["X_A"] = store["X_A"].copy()
    store["X_A"][:, names.index("cov2_ai")] = vals
    np.savez(DEV_NPZ, **store)
    print(f"{DEV_NPZ}: cov2_ai column patched", flush=True)


if __name__ == "__main__":
    main()
