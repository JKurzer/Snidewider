"""Model-collapse-inspired distributional features (single-pass, no comparisons).

The collapse literature (Shumailov et al., Curse of Recursion / Nature 2024;
Alemohammad et al., MAD) characterizes synthetic text as distributionally
CONCENTRATED: the tails thin out, rare types vanish, entropy deflates,
vocabulary growth flattens. These features distill that signature per doc —
no windows, no banks, no reference corpus.

Pure functions (RULES #5); NaN when a doc is too short.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ai_text_detection.stats_features import WORD_RE

COLLAPSE_FEATURE_NAMES = (
    "spec_k1", "spec_k2", "spec_k3", "spec_k4_7",
    "tail_mass", "heaps_25", "heaps_50", "heaps_75",
    "zipf_head_slope", "zipf_tail_slope", "slope_ratio",
    "bigram_repeat_mass", "cond_entropy",
)


def collapse_features(text: str) -> dict[str, float]:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    n = len(toks)
    nan = {k: math.nan for k in COLLAPSE_FEATURE_NAMES}
    if n < 60:
        return nan
    counts = np.array(sorted(Counter(toks).values(), reverse=True))
    v = len(counts)
    freq_of_freq = Counter(counts.tolist())

    # frequency spectrum: fraction of TYPES appearing exactly k times
    spec = {f"spec_k{k}": freq_of_freq.get(k, 0) / v for k in (1, 2, 3)}
    spec["spec_k4_7"] = sum(freq_of_freq.get(k, 0) for k in range(4, 8)) / v

    # tail mass: fraction of TOKENS from the least-frequent half of the inventory
    half = counts[v // 2:]
    tail_mass = float(half.sum() / n)

    # Heaps growth: fraction of final inventory seen at 25/50/75% of the doc
    growth = []
    seen: set[str] = set()
    marks = {n // 4, n // 2, 3 * n // 4}
    hit: dict[int, int] = {}
    for i, t in enumerate(toks):
        seen.add(t)
        if i in marks:
            hit[i] = len(seen)
    heaps = [hit[n // 4] / v, hit[n // 2] / v, hit[3 * n // 4] / v]

    # two-regime Zipf: head slope (top ~10 ranks) vs tail slope (bottom half)
    logf = np.log(counts / n)
    logr = np.log(np.arange(1, v + 1))
    h = min(10, v)
    head_slope = float(np.polyfit(logr[:h], logf[:h], 1)[0]) if h >= 3 else math.nan
    t0 = v // 2
    tail_slope = float(np.polyfit(logr[t0:], logf[t0:], 1)[0]) if v - t0 >= 3 else math.nan
    ratio = tail_slope / head_slope if head_slope and math.isfinite(head_slope) else math.nan

    # word-bigram repeat mass: fraction of tokens inside repeated bigrams
    bg = Counter(zip(toks, toks[1:]))
    repeat_mass = sum(c for c in bg.values() if c > 1) / max(1, n - 1)

    # conditional char entropy H(c2|c1) within the doc (single pass)
    chars = text.lower()
    joint = Counter(zip(chars, chars[1:]))
    first = Counter(chars[:-1])
    cond_h = 0.0
    m = max(1, len(chars) - 1)
    for (c1, c2), cnt in joint.items():
        p = cnt / m
        cond_h -= p * math.log2(cnt / first[c1])
    return {
        **spec,
        "tail_mass": tail_mass,
        "heaps_25": heaps[0], "heaps_50": heaps[1], "heaps_75": heaps[2],
        "zipf_head_slope": head_slope, "zipf_tail_slope": tail_slope,
        "slope_ratio": ratio,
        "bigram_repeat_mass": float(repeat_mass),
        "cond_entropy": cond_h,
    }
