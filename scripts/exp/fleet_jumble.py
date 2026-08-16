"""FLEET C — jumble-response of the s256-ck2 champion + coverage variants.

Base stat: W=150 random token-window pairs, CK2 metric, samples=256 (the
fleet-A2 champion). Axes:

  control   min_gap=50 on the raw text (sanity anchor; must reproduce A2)
  overlap   min_gap=0 — window pairs may overlap ("overlap the coverage")
  cascade   adaptive W in {150,100,60,30}: largest that fits ("expand the
            coverage" beyond the ~11% long-doc club)
  jXX       char-jumble of the raw text at fraction XX in {0.10, 0.11, 0.20}
            before featurizing (seeded per doc+level, jumble_gradient.py's
            scheme). The 10/11 pair probes the CROSS region (exp: humans
            shift more at f~0.01, AI more at f>=0.1).
  dYY       response deltas: base - jYY (feed-forward evidence: does the
            DEGRADATION CURVE separate classes beyond the base level?)

DEV ONLY (A/B/C). Rank on B by TPR@1e-2; AUROC direction-corrected.
Usage: .venv\\Scripts\\python scripts\\exp\\fleet_jumble.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from jumble_gradient import jumble_fraction

from ai_text_detection import burst
from ai_text_detection.evaldata import split_buckets

LEVELS = (0.10, 0.11, 0.20)
CASCADE_W = (150, 100, 60, 30)
OUT = "docs/exp/fleet_jumble.md"
FEATURES = ("control", "overlap", "cascade", "j10", "j11", "j20",
            "d10", "d11", "d20", "d10_11")


def ck2_mean(text: str, window: int, min_gap: int) -> float:
    s = burst.random_change_series(text, window=window, samples=256, min_gap=min_gap,
                                   metric="ck2", unit="tokens")
    return float(np.mean(s)) if s else float("nan")


def doc_features(text: str) -> dict[str, float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    v0 = ck2_mean(text, 150, 50)
    out = {"control": v0, "overlap": ck2_mean(text, 150, 0)}
    casc = float("nan")
    for W in CASCADE_W:
        casc = ck2_mean(text, W, 50)
        if np.isfinite(casc):
            break
    out["cascade"] = casc
    for li, lev in enumerate(LEVELS):
        out[f"j{int(lev*100):02d}"] = ck2_mean(jumble_fraction(text, lev, seed + li), 150, 50)
    for lev in LEVELS:
        tag = f"d{int(lev*100):02d}"
        out[tag] = v0 - out[f"j{int(lev*100):02d}"]
    out["d10_11"] = out["j10"] - out["j11"]
    return out


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals: dict[str, dict[str, np.ndarray]] = {}
    for b in "ABC":
        rows = [doc_features(str(t)) for t in buckets[b].generation]
        vals[b] = {f: np.array([r[f] for r in rows]) for f in FEATURES}
        print(f"bucket {b} done", flush=True)

    rows = []
    for f in FEATURES:
        row = {"feature": f}
        for b in "ABC":
            cov, roc, tpr = eval_feat(vals[b][f], labels[b])
            row[f"cov_{b}"], row[f"roc_{b}"], row[f"tpr_{b}"] = cov, roc, tpr
        rows.append(row)
    rows.sort(key=lambda r: (np.nan_to_num(r["tpr_B"]), np.nan_to_num(r["roc_B"])), reverse=True)

    lines = ["# FLEET C — jumble response of s256-ck2-mean + coverage variants\n\n",
             "base stat W150/s256/ck2/min_gap=50; jumble = seeded char-level partial shuffle; "
             "deltas = base - jumbled (response curve). DEV ONLY.\n\n",
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
