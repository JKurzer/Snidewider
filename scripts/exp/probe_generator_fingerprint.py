"""PROBE — generator fingerprinting, length-controlled + the right tail.

Two questions, purely descriptive:
  1. Do per-generator csa_wt B/char medians still separate INSIDE one length
     band? (controls the length confound from probe_csa_measure)
  2. Which generators/decodings populate the incompressible right tail
     (wt > 10 B/char in the 200-1000B band)?

Usage: .venv\\Scripts\\python scripts\\exp\\probe_generator_fingerprint.py
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
    for text, model, decoding in zip(c.generation, c.model, c.decoding):
        b = str(text).encode("utf-8")
        if len(b) < 200:
            continue
        out = _csa_native.csa_stats(b)
        rows.append({"model": model, "decoding": decoding, "n": len(b),
                     "wt": out["csa_wt_bytes"] / len(b)})
    d = pd.DataFrame(rows)

    band = d[(d.n >= 1000) & (d.n < 2000)]
    print(f"== per-generator, 1000-2000B band only (n={len(band)}) ==")
    print(f"{'model':<14} {'med':>5} {'[p25,p75]':>15}  n")
    for model, sub in sorted(band.groupby("model"),
                             key=lambda kv: kv[1].wt.median()):
        q = sub.wt.quantile([0.25, 0.5, 0.75]).values
        print(f"{model:<14} {q[1]:5.3f} [{q[0]:.3f},{q[2]:.3f}]  n={len(sub)}")

    print("\n== the incompressible tail: 200-1000B docs with wt > 10 ==")
    tail = d[(d.n < 1000) & (d.wt > 10)]
    print(f"tail size: {len(tail)} docs (of {len(d[d.n < 1000])} short docs)")
    if len(tail):
        print(tail.groupby(["model", "decoding"]).size()
              .sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
