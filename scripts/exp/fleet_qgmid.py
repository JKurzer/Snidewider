"""FLEET A — qg_mid expansion: non-mean aggregations + coverage windows.

Context: qg_mid_qgram_mean is the most holdout-robust panel feature (holdout
AUROC 0.879 solo) but covers only ~9% of dev docs (needs >= 2*W + min_gap =
350 tokens at W=150). The F3 fleet wired only mean/stdev off the random
pair-distance series; burst already computes a richer series. Hypotheses:
  1. non-mean central stats (median/p25/p75) match or beat mean (outlier-proof)
  2. smaller windows trade a little separation for a lot of coverage
  3. more samples (128 vs 32) stabilizes the tail stats

DEV ONLY — no holdout contact. Rank on B, sanity-check on A, confirm the
top rows on C. Metrics at the accepted operating point: TPR@FPR=1e-2.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_qgmid.py
"""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd

from ai_text_detection import burst
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

WINDOWS = (60, 100, 150)
METRICS = ("qgram", "ck2")
SAMPLES = 128
MIN_GAP = 50
STATS = ("mean", "median", "p10", "p25", "p75", "p90", "min", "max", "iqr", "stdev", "frac_ni")
FPR = 1e-2
OUT = "docs/exp/fleet_qgmid.md"


def series_stats(s: list[float]) -> dict[str, float]:
    if not s:
        return {k: float("nan") for k in STATS}
    dec = statistics.quantiles(s, n=10)
    q1, med, q3 = statistics.quantiles(s, n=4)
    return {
        "mean": statistics.fmean(s), "median": med, "p10": dec[0], "p25": q1,
        "p75": q3, "p90": dec[8], "min": min(s), "max": max(s), "iqr": q3 - q1,
        "stdev": statistics.pstdev(s), "frac_ni": sum(1 for v in s if v < 0.05) / len(s),
    }


def eval_feat(vals: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(coverage, oriented AUROC, TPR@1e-2). Higher score = more AI-like."""
    m = np.isfinite(vals)
    cov = float(m.mean())
    if m.sum() < 50 or len(set(y[m])) < 2:
        return (cov, float("nan"), float("nan"))
    ai, hu = vals[m][y[m] == 1], vals[m][y[m] == 0]
    raw = auroc(list(ai), list(hu))
    if raw < 0.5:
        ai, hu = -ai, -hu
        raw = 1.0 - raw
    res = tpr_at_fpr(list(ai), list(hu), fpr=FPR)
    return (cov, raw, res["tpr"])


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    names: list[str] = []
    cols: dict[str, list[np.ndarray]] = {b: [] for b in "ABC"}

    def add(feat_name: str, per_bucket: dict[str, np.ndarray]) -> None:
        names.append(feat_name)
        for b in "ABC":
            cols[b].append(per_bucket[b])

    for W in WINDOWS:
        per: dict[str, dict[str, list[float]]] = {
            b: {f"{m}_{s}": [] for m in METRICS for s in STATS} for b in "ABC"
        }
        for b in "ABC":
            for t in buckets[b].generation:
                for metric in METRICS:
                    s = burst.random_change_series(
                        str(t), window=W, samples=SAMPLES, min_gap=MIN_GAP,
                        metric=metric, unit="tokens")
                    st = series_stats(s)
                    for k in STATS:
                        per[b][f"{metric}_{k}"].append(st[k])
            print(f"W={W} bucket {b} done", flush=True)
        for metric in METRICS:
            for s in STATS:
                add(f"qg_w{W}_{metric}_{s}", {b: np.array(per[b][f"{metric}_{s}"]) for b in "ABC"})

    # incumbent baseline: the exact F3 config (W=150, samples=32, qgram mean/stdev)
    base: dict[str, dict[str, list[float]]] = {b: {"mean": [], "stdev": []} for b in "ABC"}
    for b in "ABC":
        for t in buckets[b].generation:
            bf = burst.burst_features(str(t), window=150, samples=32, min_gap=50,
                                      unit="tokens", mode="random", metric="qgram")
            base[b]["mean"].append(bf["mean"])
            base[b]["stdev"].append(bf["stdev"])
    add("INCUMBENT_qg_mid_qgram_mean", {b: np.array(base[b]["mean"]) for b in "ABC"})
    add("INCUMBENT_qg_mid_qgram_stdev", {b: np.array(base[b]["stdev"]) for b in "ABC"})

    rows = []
    for i, name in enumerate(names):
        row = {"feature": name}
        for b in "ABC":
            cov, roc, tpr = eval_feat(cols[b][i], labels[b])
            row[f"cov_{b}"], row[f"roc_{b}"], row[f"tpr_{b}"] = cov, roc, tpr
        rows.append(row)
    rows.sort(key=lambda r: (np.nan_to_num(r["roc_B"]), np.nan_to_num(r["tpr_B"])), reverse=True)

    lines = ["# FLEET A — qg_mid expansion sweep\n\n",
             f"series: random token-window pairs, samples={SAMPLES}, min_gap={MIN_GAP}; "
             f"W in {WINDOWS}; metrics {METRICS}. AUROC direction-corrected; TPR@FPR=1e-2. "
             "DEV ONLY.\n\n",
             "| feature | cov B | AUROC A | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n",
             "|---|---|---|---|---|---|---|\n"]
    for r in rows[:30]:
        lines.append(
            f"| {r['feature']} | {r['cov_B']:.3f} | {r['roc_A']:.3f} | {r['roc_B']:.3f} | "
            f"{r['roc_C']:.3f} | {r['tpr_B']:.3f} | {r['tpr_C']:.3f} |\n")
    lines.append("\n## incumbent reference\n\n")
    for r in rows:
        if r["feature"].startswith("INCUMBENT"):
            lines.append(
                f"| {r['feature']} | {r['cov_B']:.3f} | {r['roc_A']:.3f} | {r['roc_B']:.3f} | "
                f"{r['roc_C']:.3f} | {r['tpr_B']:.3f} | {r['tpr_C']:.3f} |\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
