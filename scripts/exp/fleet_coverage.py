"""FLEET F2 — reference n-gram coverage (GLTR-flavored, no LM).

feature = fraction of a doc's word q-grams present in a pooled reference set
(bucket A humans; also A AI for the contrast). Membership, not distance —
Fleet B composted the mega-profile DISTANCE; coverage is a different
statistic and gets one fair shot. References fit on A only; select on B,
confirm on C. TPR@FPR=1e-2. DEV ONLY.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_coverage.py
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat

from ai_text_detection.evaldata import split_buckets

WORD_RE = re.compile(r"[A-Za-z0-9']+")
QS = (2, 3, 5)
OUT = "docs/exp/fleet_coverage.md"


def ngrams(tokens: list[str], q: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + q]) for i in range(len(tokens) - q + 1)}


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    refs: dict[str, dict[int, set]] = {"hu": {}, "ai": {}}
    a = buckets["A"]
    for cls, sub in (("hu", a[a.model == "human"]), ("ai", a[a.model != "human"])):
        bags: dict[int, Counter] = {q: Counter() for q in QS}
        for t in sub.generation:
            toks = [w.lower() for w in WORD_RE.findall(str(t))]
            for q in QS:
                bags[q].update(ngrams(toks, q))
        refs[cls] = {q: set(bags[q]) for q in QS}
        print(f"ref {cls}: " + ", ".join(f"q{q}={len(refs[cls][q])}" for q in QS), flush=True)

    names = [f"cov{q}_hu" for q in QS] + [f"cov{q}_ai" for q in QS] + \
            [f"cov{q}_contrast" for q in QS]
    vals: dict[str, dict[str, np.ndarray]] = {}
    for b in ("B", "C"):
        out: dict[str, list[float]] = {n: [] for n in names}
        for t in buckets[b].generation:
            toks = [w.lower() for w in WORD_RE.findall(str(t))]
            cov = {}
            for q in QS:
                mine = ngrams(toks, q)
                if not mine:
                    cov[q] = (np.nan, np.nan)
                    continue
                cov[q] = (len(mine & refs["hu"][q]) / len(mine),
                          len(mine & refs["ai"][q]) / len(mine))
            for q in QS:
                out[f"cov{q}_hu"].append(cov[q][0])
                out[f"cov{q}_ai"].append(cov[q][1])
                h, a_ = cov[q]
                out[f"cov{q}_contrast"].append(a_ - h if np.isfinite(h) else np.nan)
        vals[b] = {n: np.array(v) for n, v in out.items()}
        print(f"bucket {b} scored", flush=True)

    lines = ["# FLEET F2 — reference n-gram coverage (membership, not distance)\n\n",
             "refs from A; select B, confirm C. TPR@FPR=1e-2. DEV ONLY.\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n",
             "|---|---|---|---|---|\n"]
    rows = []
    for n in names:
        rb = eval_feat(vals["B"][n], labels["B"])
        rc = eval_feat(vals["C"][n], labels["C"])
        rows.append((n, rb, rc))
    rows.sort(key=lambda r: np.nan_to_num(r[1][1]), reverse=True)
    for n, rb, rc in rows:
        lines.append(f"| {n} | {rb[1]:.3f} | {rc[1]:.3f} | {rb[2]:.3f} | {rc[2]:.3f} |\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
