"""FLEET F1 — classical statistical-detector features, solo benched.

Packs (all pure per-doc scalars, no reference data, no deps):
  zipf       rank-frequency OLS on log-log: slope/intercept/R2 + top10 mass,
             hapax/dis ratios (Zipf's law is THE classical detector stat)
  richness   TTR, Guiraud R, Herdan C, Yule K, Simpson D, Maas a2 (same Counter)
  readability sentence/word length stats, Flesch (vowel-group syllable
             heuristic), punctuation/digit/uppercase/contraction rates
  compress   zlib/bz2 compression ratio + self-concatenation gain

DEV ONLY. Rank on B by AUROC, confirm on C; TPR@FPR=1e-2. NaN-safe.
Usage: .venv\\Scripts\\python scripts\\exp\\fleet_stats.py
"""

from __future__ import annotations

import bz2
import math
import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat

from ai_text_detection.evaldata import split_buckets

WORD_RE = re.compile(r"[A-Za-z0-9']+")
OUT = "docs/exp/fleet_stats.md"


def zipf_richness(tokens: list[str]) -> dict[str, float]:
    n = len(tokens)
    if n < 30:
        return {k: math.nan for k in (
            "zipf_slope", "zipf_r2", "zipf_top10", "hapax", "dis", "ttr",
            "guiraud", "herdan_c", "yule_k", "simpson_d", "maas_a2")}
    counts = np.array(sorted(Counter(t.lower() for t in tokens).values(), reverse=True))
    v = len(counts)
    freqs = counts / n
    rank = np.arange(1, v + 1)
    if v >= 10:
        A = np.vstack([np.log(rank), np.ones(v)]).T
        slope, intercept = np.linalg.lstsq(A, np.log(freqs), rcond=None)[0]
        pred = A @ [slope, intercept]
        ss_res = float(((np.log(freqs) - pred) ** 2).sum())
        ss_tot = float(((np.log(freqs) - np.log(freqs).mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else math.nan
    else:
        slope, r2 = math.nan, math.nan
    freq_of_freq = Counter(counts.tolist())
    hapax = freq_of_freq.get(1, 0) / v
    dis = freq_of_freq.get(2, 0) / v
    yule = 1e4 * (float((counts**2 * [freq_of_freq.get(int(f), 0) for f in counts]).sum()) - n) / (n * n)
    simpson = float((counts * (counts - 1)).sum()) / (n * (n - 1))
    return {
        "zipf_slope": float(slope), "zipf_r2": r2,
        "zipf_top10": float(counts[:10].sum() / n),
        "hapax": hapax, "dis": dis, "ttr": v / n,
        "guiraud": v / math.sqrt(n), "herdan_c": math.log(v) / math.log(n),
        "yule_k": yule, "simpson_d": simpson,
        "maas_a2": (math.log(n) - math.log(v)) / (math.log(n) ** 2),
    }


def readability(text: str, tokens: list[str]) -> dict[str, float]:
    sents = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 2]
    slen = [len(s.split()) for s in sents]
    wlen = [len(t) for t in tokens] or [0]
    syll = sum(max(1, len(re.findall(r"[aeiouy]+", t.lower()))) for t in tokens)
    n_tok = max(1, len(tokens))
    words_per_sent = n_tok / max(1, len(slen))
    return {
        "sent_len_mean": float(np.mean(slen)) if slen else math.nan,
        "sent_len_stdev": float(np.std(slen)) if len(slen) > 1 else math.nan,
        "word_len_mean": float(np.mean(wlen)),
        "word_len_stdev": float(np.std(wlen)),
        "flesch": 206.835 - 1.015 * words_per_sent - 84.6 * (syll / n_tok),
        "punct_rate": sum(1 for c in text if c in ".,;:!?—-()\"'") / max(1, len(text)),
        "digit_rate": sum(1 for c in text if c.isdigit()) / max(1, len(text)),
        "contraction_rate": sum(1 for t in tokens if "'" in t) / n_tok,
    }


def compressibility(text: str) -> dict[str, float]:
    raw = text.encode("utf-8")
    n = max(1, len(raw))
    z = len(zlib.compress(raw, 9)) / n
    b = len(bz2.compress(raw, 9)) / n
    z2 = len(zlib.compress(raw + raw, 9)) / (2 * n)
    return {"zlib_ratio": z, "bz2_ratio": b, "zlib_selfgain": (z - z2) / z if z else math.nan}


def doc_features(text: str) -> dict[str, float]:
    tokens = WORD_RE.findall(text)
    out = {}
    out.update(zipf_richness(tokens))
    out.update(readability(text, tokens))
    out.update(compressibility(text))
    return out


FEATS = list(doc_features("sample text. another one here! " * 10).keys())


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals: dict[str, dict[str, np.ndarray]] = {}
    for b in "ABC":
        rows = [doc_features(str(t)) for t in buckets[b].generation]
        vals[b] = {f: np.array([r[f] for r in rows]) for f in FEATS}
        print(f"bucket {b} done", flush=True)

    rows = []
    for f in FEATS:
        row = {"feature": f}
        for b in "ABC":
            cov, roc, tpr = eval_feat(vals[b][f], labels[b])
            row[f"cov_{b}"], row[f"roc_{b}"], row[f"tpr_{b}"] = cov, roc, tpr
        rows.append(row)
    rows.sort(key=lambda r: np.nan_to_num(r["roc_B"]), reverse=True)

    lines = ["# FLEET F1 — classical statistical-detector features\n\n",
             "zipf/richness/readability/compress packs. DEV ONLY. TPR@FPR=1e-2.\n\n",
             "| feature | AUROC A | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n",
             "|---|---|---|---|---|---|\n"]
    for r in rows:
        lines.append(
            f"| {r['feature']} | {r['roc_A']:.3f} | {r['roc_B']:.3f} | {r['roc_C']:.3f} | "
            f"{r['tpr_B']:.3f} | {r['tpr_C']:.3f} |\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
