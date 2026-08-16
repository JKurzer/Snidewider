"""FLEET B — <test> vs <general>: pooled mega-corpus reference features.

The idea (old Donk design): instead of distance-to-individual-exemplars
(exemplar.py, <test> vs <test_texts>), pool ALL of bucket A into one mega
q-gram profile per class and score <test> vs <general>. The per-doc bank
stats (min/p10) ride a handful of nearest exemplars; the mega profile
measures typicality against the whole class distribution — no nearest-
neighbor luck involved, and it's ONE native diff per side.

Leakage discipline: references built from bucket A only; selection on B;
confirmation on C. Calibration percentile is fit on B humans (A humans sit
inside the mega reference — leave-one-out impossible) and read on C.
Novelty: Spearman vs the existing ex_* panel columns on C.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_humanbank.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import qgram
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

QS = (3, 5)
FPR = 1e-2
OUT = "docs/exp/fleet_humanbank.md"
PANEL = ("ex_hu_mean", "ex_hu_min", "ex_contrast_p10", "qg_mid_qgram_mean")


def profile_total(profile: qgram.Profile) -> int:
    return sum(c for _, c in profile)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    a = buckets["A"]
    mega = {}
    for cls, sub in (("hu", a[a.model == "human"]), ("ai", a[a.model != "human"])):
        blob = "\n\n".join(str(t) for t in sub.generation).encode("utf-8")
        mega[cls] = {q: (qgram.profile(blob, q), None) for q in QS}
        for q in QS:
            mega[cls][q] = (mega[cls][q][0], profile_total(mega[cls][q][0]))
        print(f"mega {cls}: {len(sub)} docs, blob {len(blob)} bytes", flush=True)

    names: list[str] = []
    feats: dict[str, dict[str, np.ndarray]] = {}

    def score_bucket(b: str) -> dict[str, np.ndarray]:
        out: dict[str, list[float]] = {n: [] for n in names}
        for t in buckets[b].generation:
            blob = str(t).encode("utf-8")
            for q in QS:
                dp = qgram.profile(blob, q)
                dtot = max(0, len(blob) - q + 1)
                d = {}
                for cls in ("hu", "ai"):
                    prof, mtot = mega[cls][q]
                    raw = qgram.distance_profiles(dp, prof)
                    d[cls] = raw / (dtot + mtot) if dtot + mtot else 0.0
                out[f"q{q}_d_hu"].append(d["hu"])
                out[f"q{q}_d_ai"].append(d["ai"])
                out[f"q{q}_contrast"].append(d["ai"] - d["hu"])
                out[f"q{q}_ratio"].append(d["hu"] / (d["hu"] + d["ai"]) if d["hu"] + d["ai"] else 0.5)
        return {k: np.array(v) for k, v in out.items()}

    names.extend(f"q{q}_{s}" for q in QS for s in ("d_hu", "d_ai", "contrast", "ratio"))
    for b in ("B", "C"):
        feats[b] = score_bucket(b)
        print(f"bucket {b} scored", flush=True)

    # calibration percentile: fit on B humans (A is inside the reference), read on C
    for q in QS:
        ref = np.sort(feats["B"][f"q{q}_d_hu"][labels["B"] == 0])
        for b in ("B", "C"):
            vals = feats[b][f"q{q}_d_hu"]
            feats[b][f"q{q}_calib_pct"] = np.searchsorted(ref, vals) / len(ref)
        names.append(f"q{q}_calib_pct")

    def eval_feat(vals: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        ai, hu = vals[y == 1], vals[y == 0]
        raw = auroc(list(ai), list(hu))
        if raw < 0.5:
            ai, hu = -ai, -hu
            raw = 1.0 - raw
        return raw, tpr_at_fpr(list(ai), list(hu), fpr=FPR)["tpr"]

    # novelty: spearman vs existing panel columns on C (name-matched)
    panel = np.load("data/derived/full_features.npz")
    panel_names = list(panel["feature_names"])
    panel_idx = {n: panel_names.index(n) for n in PANEL if n in panel_names}
    Xc, yc = panel["X_C"], panel["y_C"]

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    lines = ["# FLEET B — <test> vs <general> mega-reference sweep\n\n",
             "references: pooled bucket-A profiles per class (one native diff per side). "
             "selection on B, confirmation on C; calib fit on B humans. TPR@FPR=1e-2. DEV ONLY.\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C | max |rho| vs panel |\n",
             "|---|---|---|---|---|---|\n"]
    rows = []
    for n in names:
        roc_b, tpr_b = eval_feat(feats["B"][n], labels["B"])
        roc_c, tpr_c = eval_feat(feats["C"][n], labels["C"])
        rho = max((abs(spearman(feats["C"][n], Xc[:, idx])) for idx in panel_idx.values()),
                  default=float("nan"))
        rows.append((n, roc_b, roc_c, tpr_b, tpr_c, rho))
    rows.sort(key=lambda r: r[2], reverse=True)
    for n, rb, rc, tb, tc, rho in rows:
        lines.append(f"| {n} | {rb:.3f} | {rc:.3f} | {tb:.3f} | {tc:.3f} | {rho:.2f} |\n")

    lines.append("\n## panel reference on C (existing features)\n\n")
    for pname, idx in panel_idx.items():
        roc_c, tpr_c = eval_feat(Xc[:, idx], yc)
        lines.append(f"- {pname}: AUROC {roc_c:.3f}, TPR@1e-2 {tpr_c:.3f}\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
