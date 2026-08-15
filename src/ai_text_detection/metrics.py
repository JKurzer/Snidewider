"""Evaluation metrics shared by scripts and the future eval harness."""

import math


def auroc(ai_scores: list[float], human_scores: list[float]) -> float:
    """P(ai > human) via the Mann-Whitney rank statistic.

    Direction lives in the value: >0.5 means AI scores higher than human.
    Ties get average ranks — matters for discrete features (e.g. q-gram
    count percentiles), where tuple-order tie-breaking would fabricate AUROC.
    """
    n_ai, n_h = len(ai_scores), len(human_scores)
    if n_ai == 0 or n_h == 0:
        raise ValueError("both classes must be non-empty")
    combined = sorted([(s, 1) for s in ai_scores] + [(s, 0) for s in human_scores])
    rank_sum_ai = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # 1-based mean rank of the tie block
        rank_sum_ai += avg_rank * sum(is_ai for _, is_ai in combined[i : j + 1])
        i = j + 1
    return (rank_sum_ai - n_ai * (n_ai + 1) / 2) / (n_ai * n_h)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def tpr_at_fpr(
    ai_scores: list[float], human_scores: list[float], fpr: float = 1e-3
) -> dict[str, float]:
    """TPR at a fixed FPR with a Wilson CI. Higher score = more AI-like.

    Threshold is the (1 - fpr) quantile of human scores (nearest-rank).
    """
    if not ai_scores or not human_scores:
        raise ValueError("both classes must be non-empty")
    ordered = sorted(human_scores)
    rank = max(0, math.ceil((1 - fpr) * len(ordered)) - 1)
    threshold = ordered[rank]
    hits = sum(1 for s in ai_scores if s >= threshold)
    lo, hi = wilson_ci(hits, len(ai_scores))
    return {
        "threshold": threshold,
        "tpr": hits / len(ai_scores),
        "tpr_lo": lo,
        "tpr_hi": hi,
        "n_ai": float(len(ai_scores)),
        "n_human": float(len(human_scores)),
    }
