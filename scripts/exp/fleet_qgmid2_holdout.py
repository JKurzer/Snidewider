"""Holdout confirmation — qg_mid expansion champion, PRE-DECLARED and minimal.

Selection was entirely dev-side (fleet_qgmid/fleet_qgmid2): W=150 token
random pairs, min_gap=50, CK2 metric, samples=256. This script reads holdout
for exactly two pre-declared variants, no more:

  1. qg_w150_s256_ck2_mean   (dev C: AUROC 0.985, TPR@1e-2 0.767, cov 0.111)
  2. qg_w150_s256_ck2_median (dev C: AUROC 0.987, TPR@1e-2 0.744, cov 0.111)

Reference points already on record: incumbent qg_mid_qgram_mean holdout AUROC
0.879 / qg_mid_ck2_mean 0.765 (docs/holdout_breakout.md, samples=32 config).
No re-selection happens here — this is the once-clean confirmation read.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_qgmid2_holdout.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import burst
from ai_text_detection.metrics import auroc, tpr_at_fpr

OUT = "docs/exp/fleet_qgmid2_holdout.md"


def series_mean_median(text: str) -> tuple[float, float]:
    s = burst.random_change_series(text, window=150, samples=256, min_gap=50,
                                   metric="ck2", unit="tokens")
    if not s:
        return (float("nan"), float("nan"))
    return (float(np.mean(s)), float(np.median(s)))


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    hold = df[df.fold == "holdout"]
    subs = {"human": hold[hold.model == "human"],
            "ai": hold[hold.model != "human"].sample(n=20_000, random_state=97)}
    vals: dict[str, np.ndarray] = {}
    for name, sub in subs.items():
        rows = [series_mean_median(str(t)) for t in sub.generation]
        vals[name] = np.array(rows)
        print(f"holdout {name}: {len(rows)} docs scored", flush=True)

    lines = ["# Holdout confirmation — qg_mid expansion champion (pre-declared)\n\n",
             "Variants fixed on dev BEFORE this read: W150/s256/ck2 mean + median. "
             "Metrics on covered docs only (NaN = doc too short). \n\n",
             "| variant | coverage hu/ai | AUROC | TPR@5e-2 | TPR@1e-2 | TPR@1e-3 |\n",
             "|---|---|---|---|---|---|\n"]
    for j, variant in ((0, "mean"), (1, "median")):
        hu, ai = vals["human"][:, j], vals["ai"][:, j]
        mh, ma = np.isfinite(hu), np.isfinite(ai)
        # orientation: dev showed AI scores LOWER (ai>lower) — flip to higher=AI
        ai_s, hu_s = -ai[ma], -hu[mh]
        roc = auroc(list(ai_s), list(hu_s))
        cells = []
        for fpr in (5e-2, 1e-2, 1e-3):
            r = tpr_at_fpr(list(ai_s), list(hu_s), fpr=fpr)
            cells.append(f"{r['tpr']:.3f} [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}]")
        lines.append(f"| {variant} | {mh.mean():.3f}/{ma.mean():.3f} | {roc:.4f} | "
                     + " | ".join(cells) + " |\n")
        print(lines[-1], flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
