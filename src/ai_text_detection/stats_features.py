"""Classical statistical-detector features (fleet F1, promoted to the panel).

Four packs, all pure per-doc scalars with no reference data and no deps:
  zipf       rank-frequency OLS on log-log (slope/R2), top-10 mass, hapax/dis
  richness   TTR, Guiraud R, Herdan C, Yule K, Simpson D, Maas a2
  readability sentence/word length stats, Flesch (vowel-group syllables),
             punctuation/digit/contraction rates
  compress   zlib/bz2 compression ratio + self-concatenation gain

Pure functions (RULES #5). NaN when a doc is too short for a stat.
"""

from __future__ import annotations

import bz2
import math
import re
import zlib
from collections import Counter

import numpy as np

WORD_RE = re.compile(r"[A-Za-z0-9']+")

STAT_FEATURE_NAMES = (
    "zipf_slope", "zipf_r2", "zipf_top10", "hapax", "dis", "ttr",
    "guiraud", "herdan_c", "yule_k", "simpson_d", "maas_a2",
    "sent_len_mean", "sent_len_stdev", "word_len_mean", "word_len_stdev",
    "flesch", "punct_rate", "digit_rate", "contraction_rate",
    "zlib_ratio", "bz2_ratio", "zlib_selfgain",
)


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


def stat_features(text: str) -> dict[str, float]:
    """All 22 classical stats for one doc (order = STAT_FEATURE_NAMES)."""
    tokens = WORD_RE.findall(text)
    out: dict[str, float] = {}
    out.update(zipf_richness(tokens))
    out.update(readability(text, tokens))
    out.update(compressibility(text))
    return out
