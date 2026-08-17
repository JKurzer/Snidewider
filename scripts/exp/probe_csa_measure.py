"""PROBE — what does CSA compressed size (B/char) look like for AI vs human?

NOT a detector readout: no thresholds, no metrics. Distributional shape only,
controlled for doc length (CSA size has fixed overhead that amortizes).

Usage: .venv\\Scripts\\python scripts\\exp\\probe_csa_measure.py
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
        rows.append({"model": model, "n": len(b),
                     "wt": out["csa_wt_bytes"] / len(b),
                     "sada": out["csa_sada_bytes"] / len(b)})
    d = pd.DataFrame(rows)
    print(f"n docs: {len(d)} (humans {(d.model == 'human').sum()}, ai {(d.model != 'human').sum()})\n")

    def band(n: int) -> str:
        for lo, hi in ((200, 1000), (1000, 2000), (2000, 4000), (4000, 10**9)):
            if lo <= n < hi:
                return f"{lo}-{hi if hi < 10**9 else 'up'}"
        return "?"

    d["lenband"] = d.n.map(band)
    d["cls"] = np.where(d.model == "human", "human", "ai")

    print("== csa_wt B/char: class x length band ==  (median [p25,p75], n)")
    for bandname in ("200-1000", "1000-2000", "2000-4000", "4000-up"):
        sub = d[d.lenband == bandname]
        if len(sub) == 0:
            continue
        for cls in ("human", "ai"):
            s = sub[sub.cls == cls]
            if len(s) < 10:
                continue
            q = s.wt.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
            print(f"  {bandname:>9} {cls:<6} med {q[2]:.3f} [{q[1]:.3f},{q[3]:.3f}] "
                  f"p05 {q[0]:.3f} p95 {q[4]:.3f}  n={len(s)}")

    print("\n== same for csa_sada B/char ==")
    for bandname in ("200-1000", "1000-2000", "2000-4000", "4000-up"):
        sub = d[d.lenband == bandname]
        for cls in ("human", "ai"):
            s = sub[(sub.cls == cls) & (sub.lenband == bandname)]
            if len(s) < 10:
                continue
            q = s.sada.quantile([0.25, 0.5, 0.75]).values
            print(f"  {bandname:>9} {cls:<6} med {q[1]:.3f} [{q[0]:.3f},{q[2]:.3f}]  n={len(s)}")

    print("\n== per-generator (all lengths, wt B/char median [p25,p75], n) ==")
    for model, sub in d.groupby("model"):
        if len(sub) < 10:
            continue
        q = sub.wt.quantile([0.25, 0.5, 0.75]).values
        print(f"  {model:<14} med {q[1]:.3f} [{q[0]:.3f},{q[2]:.3f}]  n={len(sub)}")




if __name__ == "__main__":
    main()
