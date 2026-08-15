"""Document feature extraction: three dumb-signal families, pure functions.

Families (each plausibly captures a different signal):
  short_range_*  — adjacent 64-byte window CK2 distance (local surface change)
  midrange_*     — random long-range token-window pairs (smoothing breakdown)
  qgram_*        — co-occurrence stats of the doc's q-gram profile (repetition)

All NaN-free only for docs long enough to support the windows; short docs get
NaNs and the caller decides (impute, drop, or route to a short-doc model).
"""

from __future__ import annotations

import math

from ai_text_detection import burst, qgram

FEATURE_NAMES = (
    "short_range_mean",
    "short_range_stdev",
    "midrange_mean",
    "midrange_stdev",
    "qgram_total",
    "qgram_distinct_ratio",
    "qgram_repeat_frac",
    "qgram_max_share",
    "qgram_entropy",
)


def qgram_profile_features(text: str, q: int = 3) -> dict[str, float]:
    """Co-occurrence statistics of the document's q-gram profile."""
    profile = qgram.profile(text.encode("utf-8"), q)
    total = sum(c for _, c in profile)
    if total == 0:
        return {
            "qgram_total": 0.0,
            "qgram_distinct_ratio": math.nan,
            "qgram_repeat_frac": math.nan,
            "qgram_max_share": math.nan,
            "qgram_entropy": math.nan,
        }
    distinct = len(profile)
    repeated = sum(1 for _, c in profile if c > 1)
    max_count = max(c for _, c in profile)
    entropy = -sum((c / total) * math.log2(c / total) for _, c in profile)
    return {
        "qgram_total": float(total),
        "qgram_distinct_ratio": distinct / total,
        "qgram_repeat_frac": repeated / distinct,
        "qgram_max_share": max_count / total,
        "qgram_entropy": entropy,
    }


def document_features(text: str) -> dict[str, float]:
    """The full tiny-classifier feature vector. Pure; order = FEATURE_NAMES."""
    short = burst.burst_features(text, window=64, gap=0, unit="bytes")
    mid = burst.burst_features(
        text, window=150, samples=32, min_gap=50, unit="tokens", mode="random"
    )
    return {
        "short_range_mean": short["mean"],
        "short_range_stdev": short["stdev"],
        "midrange_mean": mid["mean"],
        "midrange_stdev": mid["stdev"],
        **qgram_profile_features(text),
    }


def feature_vector(text: str) -> list[float]:
    feats = document_features(text)
    return [feats[name] for name in FEATURE_NAMES]
