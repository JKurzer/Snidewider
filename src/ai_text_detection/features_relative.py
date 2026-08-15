"""Doc-length-relative window features: same signals as features.py, zero drops.

features.py uses absolute windows (64B short-range, 150-token midrange), so any
doc too short for the windows gets NaN and is dropped — 4869/6000 RAID dev docs
in the baseline eval, a massive length bias. Here the windows scale with the
document instead:

  short_range  — 64B windows only if the doc is >= 512B, else n_bytes//8
  midrange     — window = clamp(n_tokens//6, 16, 150), min_gap = max(n_tokens//12, 8)

Both shrink-to-fit (halving) for pathologically tiny docs rather than NaN-ing.
qgram_total is dropped: it is a pure document-length proxy, and RAID AI docs
are systematically shorter than human ones, so it confounds every classifier
that touches it. The remaining qgram stats are ratios/entropies — already
length-normalized.

Pure functions (RULES #5), same metrics/families as features.py.
"""

from __future__ import annotations

from ai_text_detection import burst
from ai_text_detection.features import qgram_profile_features

FEATURE_NAMES_RELATIVE = (
    "short_range_mean",
    "short_range_stdev",
    "midrange_mean",
    "midrange_stdev",
    "qgram_distinct_ratio",
    "qgram_repeat_frac",
    "qgram_max_share",
    "qgram_entropy",
)


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _fit_pair(n: int, window: int, min_gap: int) -> tuple[int, int]:
    """Shrink (window, then min_gap) until 2*window + min_gap fits in n units.

    random_change_series returns [] (→ NaN stats) unless n >= 2*window+min_gap.
    Degrading the probe beats dropping the doc.
    """
    while window > 1 and 2 * window + min_gap > n:
        window = max(1, window // 2)
    if 2 * window + min_gap > n:
        min_gap = max(0, n - 2 * window)
    return window, min_gap


def short_range_window(n_bytes: int) -> int:
    """64B windows for docs >= 512B, else n_bytes//8 (clamped), shrink-to-fit."""
    window = 64 if n_bytes >= 512 else _clamp(n_bytes // 8, 8, 64)
    while window > 1 and 2 * window > n_bytes:  # step mode, gap=0
        window //= 2
    return window


def midrange_params(n_tokens: int) -> tuple[int, int]:
    """window = clamp(n_tokens//6, 16, 150); min_gap = max(n_tokens//12, 8)."""
    window = _clamp(n_tokens // 6, 16, 150)
    min_gap = max(n_tokens // 12, 8)
    return _fit_pair(n_tokens, window, min_gap)


def document_features_relative(text: str) -> dict[str, float]:
    """Relative-window feature vector. Pure; order = FEATURE_NAMES_RELATIVE.

    NaN only for degenerate inputs (<2 tokens / <2 bytes / <3-byte qgram
    profile); every real doc yields a full vector.
    """
    n_tokens = len(text.split())
    n_bytes = len(text.encode("utf-8"))
    short_w = short_range_window(n_bytes)
    mid_w, mid_gap = midrange_params(n_tokens)
    short = burst.burst_features(text, window=short_w, gap=0, unit="bytes")
    mid = burst.burst_features(
        text, window=mid_w, samples=32, min_gap=mid_gap, unit="tokens", mode="random"
    )
    qg = qgram_profile_features(text)
    return {
        "short_range_mean": short["mean"],
        "short_range_stdev": short["stdev"],
        "midrange_mean": mid["mean"],
        "midrange_stdev": mid["stdev"],
        "qgram_distinct_ratio": qg["qgram_distinct_ratio"],
        "qgram_repeat_frac": qg["qgram_repeat_frac"],
        "qgram_max_share": qg["qgram_max_share"],
        "qgram_entropy": qg["qgram_entropy"],
    }


def feature_vector_relative(text: str) -> list[float]:
    feats = document_features_relative(text)
    return [feats[name] for name in FEATURE_NAMES_RELATIVE]
