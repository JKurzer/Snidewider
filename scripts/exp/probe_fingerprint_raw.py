"""PROBE — the raw generator fingerprint: length habit x compressibility habit.

No controls. The pooled per-generator picture as measured: median doc length
next to median csa_wt B/char. Length is a MEDIATOR here — part of the
fingerprint, not a confound to sterilize.

Usage: .venv\\Scripts\\python scripts\\exp\\probe_fingerprint_raw.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import _csa_native
from ai_text_detection.evaldata import split_buckets


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    c = split_buckets(df)["C"]

    rows = []
    for text, model in zip(c.generation, c.model):
        b = str(text).encode("utf-8")
        if len(b) < 200:
            continue
        out = _csa_native.csa_stats(b)
        rows.append({"model": model, "n": len(b), "wt": out["csa_wt_bytes"] / len(b)})
    d = pd.DataFrame(rows)

    print(f"{'model':<14} {'med_len':>7} {'wt med':>7} {'[p25,p75]':>15}  n")
    for model, sub in sorted(d.groupby("model"), key=lambda kv: kv[1].wt.median()):
        q = sub.wt.quantile([0.25, 0.5, 0.75]).values
        print(f"{model:<14} {int(sub.n.median()):>7} {q[1]:>7.3f} "
              f"[{q[0]:.3f},{q[2]:.3f}]  n={len(sub)}")


if __name__ == "__main__":
    main()
