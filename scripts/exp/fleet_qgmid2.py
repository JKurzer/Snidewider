"""FLEET A2 — long-window follow-up: Fleet A says separation grows with W
(60 < 100 < 150) and CK2 owns the tail. Push the flank: W in {150, 200, 300},
samples {128, 256}, ck2 only, central stats. DEV ONLY. TPR@FPR=1e-2.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_qgmid2.py   (run from repo root)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fleet_qgmid import eval_feat, series_stats

from ai_text_detection import burst
from ai_text_detection.evaldata import split_buckets

WINDOWS = (150, 200, 300)
SAMPLES = (128, 256)
STATS = ("mean", "median", "p25")
FPR = 1e-2
OUT = "docs/exp/fleet_qgmid2.md"


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    rows = []
    for W in WINDOWS:
        for M in SAMPLES:
            per = {b: {s: [] for s in STATS} for b in "ABC"}
            for b in "ABC":
                for t in buckets[b].generation:
                    s = burst.random_change_series(
                        str(t), window=W, samples=M, min_gap=50,
                        metric="ck2", unit="tokens")
                    st = series_stats(s)
                    for k in STATS:
                        per[b][k].append(st[k])
                print(f"W={W} M={M} bucket {b} done", flush=True)
            for s in STATS:
                row = {"feature": f"qg_w{W}_s{M}_ck2_{s}"}
                for b in "ABC":
                    cov, roc, tpr = eval_feat(np.array(per[b][s]), labels[b])
                    row[f"cov_{b}"], row[f"roc_{b}"], row[f"tpr_{b}"] = cov, roc, tpr
                rows.append(row)
    rows.sort(key=lambda r: (np.nan_to_num(r["roc_B"]), np.nan_to_num(r["tpr_B"])), reverse=True)

    lines = ["# FLEET A2 — long-window ck2 sweep (samples x windows)\n\n",
             "| feature | cov B | AUROC A | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n",
             "|---|---|---|---|---|---|---|\n"]
    for r in rows:
        lines.append(
            f"| {r['feature']} | {r['cov_B']:.3f} | {r['roc_A']:.3f} | {r['roc_B']:.3f} | "
            f"{r['roc_C']:.3f} | {r['tpr_B']:.3f} | {r['tpr_C']:.3f} |\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
