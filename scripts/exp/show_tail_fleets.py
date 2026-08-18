"""Long-tail (TPR@1e-3) solo read for fleets T (jumble-gradient), U (ApEn),
V (jumble contrast) on dev B and C. Recomputes feature values; orients by
median; reports the deep tail per feature per bucket.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fleet_apen import FEATS as U_FEATS, apen_features  # noqa: E402
from fleet_jumble_contrast import FEATS as V_FEATS, jc_features  # noqa: E402
from fleet_zipf_slope import FEATS as T_FEATS, js_features  # noqa: E402

from ai_text_detection.evaldata import split_buckets  # noqa: E402
from ai_text_detection.metrics import tpr_at_fpr  # noqa: E402


def tail_at(vals: np.ndarray, y: np.ndarray, fpr: float) -> float:
    m = np.isfinite(vals)
    if m.sum() < 100:
        return float("nan")
    pos, neg = vals[m & (y == 1)], vals[m & (y == 0)]
    r = tpr_at_fpr(list(pos), list(neg), fpr=fpr)
    r_flip = tpr_at_fpr(list(-pos), list(-neg), fpr=fpr)
    return max(r["tpr"], r_flip["tpr"])


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    packs = {
        "T zipf-gradient": (T_FEATS, lambda t, i: js_features(t, 10_000 + i)),
        "U apen": (U_FEATS, lambda t, i: apen_features(t)),
        "V jumble-contrast": (V_FEATS, lambda t, i: jc_features(t, 20_000 + i)),
    }

    for pack, (feats, fn) in packs.items():
        print(f"\n=== {pack} — TPR@1e-3 (B | C) ===", flush=True)
        cols = {b: np.array([[fn(str(t), i)[k] for k in feats]
                             for i, t in enumerate(buckets[b].generation)])
                for b in "BC"}
        for j, f in enumerate(feats):
            tb = tail_at(cols["B"][:, j], labels["B"], 1e-3)
            tc = tail_at(cols["C"][:, j], labels["C"], 1e-3)
            print(f"  {f:<22} {tb:.3f} | {tc:.3f}", flush=True)


if __name__ == "__main__":
    main()
