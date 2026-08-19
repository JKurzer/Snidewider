"""Self-anchor features (colinear-chaining lineage, Mäkinen et al.).

Exact-match anchors of the doc against ITSELF (repeated k-grams) as
content-addressed measurement sites. Unlike random window sampling, anchors
latch STRUCTURE: every doc's features answer the same question - "how does
this doc behave at its own repeat sites?" - so the channel generalizes
across rows (the raw256 lesson).

Features:
  anc_n_repeats    distinct repeated k-grams per KB
  anc_coverage     byte fraction inside any repeated k-gram
  anc_chain_rate   longest colinear chain / anchor sites (repeat coherence)
  anc_ctx_*        CK2 context-similarity distribution at anchor sites
                   (mean/stdev/p10/p90) - verbatim-context fidelity
  anc_ctx_frac_hi  fraction of anchor pairs with CK2 > 0.5

Pure functions (RULES #5); NaN when too short.
"""

from __future__ import annotations

import math

import numpy as np

from ai_text_detection import ck2

ANCHOR_FEATURE_NAMES = (
    "anc_n_repeats", "anc_coverage", "anc_chain_rate",
    "anc_ctx_mean", "anc_ctx_stdev", "anc_ctx_p10", "anc_ctx_p90",
    "anc_ctx_frac_hi",
)

K = 8            # anchor width in bytes
MAX_OCC = 30     # skip boilerplate grams with more occurrences
MAX_PAIRS = 64   # cap anchor pairs scored (perf)
CTX = 64         # context window around each occurrence


def _anchor_positions(b: bytes) -> dict[bytes, list[int]]:
    pos: dict[bytes, list[int]] = {}
    for i in range(len(b) - K + 1):
        g = b[i:i + K]
        if g in pos:
            if len(pos[g]) < MAX_OCC:
                pos[g].append(i)
        else:
            pos[g] = [i]
    return {g: ps for g, ps in pos.items() if len(ps) >= 2}


def _colinear_chain(pairs: list[tuple[int, int]]) -> int:
    """Longest chain of pairs colinear on both coordinates (LIS on p2 after
    sorting by p1). The repeat skeleton's coherence."""
    if not pairs:
        return 0
    pairs = sorted(set(pairs))
    tails: list[int] = []
    for _, p2 in pairs:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < p2:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(p2)
        else:
            tails[lo] = p2
    return len(tails)


def anchor_features(text: str) -> dict[str, float]:
    b = text.encode("utf-8")
    n = len(b)
    if n < 4 * K + CTX:
        return {k: math.nan for k in ANCHOR_FEATURE_NAMES}

    anchors = _anchor_positions(b)
    reps = len(anchors)

    covered = np.zeros(n, dtype=bool)
    pairs: list[tuple[int, int]] = []
    for ps in anchors.values():
        for p in ps:
            covered[p:p + K] = True
        pairs.extend((ps[i], ps[j]) for i in range(len(ps))
                     for j in range(i + 1, len(ps)))
    coverage = float(covered.mean())

    ctx_scores = []
    for p1, p2 in pairs[:MAX_PAIRS]:
        w1 = b[max(0, p1 - CTX):p1 + K + CTX]
        w2 = b[max(0, p2 - CTX):p2 + K + CTX]
        ctx_scores.append(ck2.similarity(w1, w2))
    ctx = np.array(ctx_scores) if ctx_scores else np.array([np.nan])

    return {
        "anc_n_repeats": reps / (n / 1000),
        "anc_coverage": coverage,
        "anc_chain_rate": _colinear_chain(pairs) / max(1, len(pairs)),
        "anc_ctx_mean": float(np.nanmean(ctx)),
        "anc_ctx_stdev": float(np.nanstd(ctx)),
        "anc_ctx_p10": float(np.nanpercentile(ctx, 10)),
        "anc_ctx_p90": float(np.nanpercentile(ctx, 90)),
        "anc_ctx_frac_hi": float(np.nanmean(ctx > 0.5)),
    }
