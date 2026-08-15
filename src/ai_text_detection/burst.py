"""Local-change ("burstiness") features from windowed self-comparison.

Strategy: take a window of W tokens, step forward K, compare the two windows.
Repeat down the document. The resulting series measures how much the document
changes token-by-token within a window — AI text tends to be more locally
uniform (less bursty) than human text.

Pure functions (RULES #5): no hidden state, no caching surprises. Distances
are normalized to [0, 1] with 0.0 = identical windows.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics

from ai_text_detection import ck2, qgram


def _window_bytes(tokens: list[str], start: int, window: int) -> bytes:
    return " ".join(tokens[start : start + window]).encode("utf-8")


def _distance(a: bytes, b: bytes, metric: str) -> float:
    if metric == "ck2":
        return ck2.similarity(a, b)
    if metric == "qgram":
        return 1.0 - qgram.similarity(a, b)
    if metric == "bag":
        return 1.0 - qgram.bag_similarity(a, b)
    raise ValueError(f"unknown metric: {metric!r}")


def _make_getter(text: str, unit: str, window: int):
    if unit == "bytes":
        raw = text.encode("utf-8")
        return lambda start: raw[start : start + window], len(raw)
    if unit == "tokens":
        parts = text.split()
        return lambda start: " ".join(parts[start : start + window]).encode("utf-8"), len(parts)
    raise ValueError(f"unknown unit: {unit!r}")


def change_series(
    text: str,
    window: int = 20,
    gap: int = 0,
    step: int | None = None,
    metric: str = "ck2",
    unit: str = "tokens",
) -> list[float]:
    """Per-step distance between windows [i, i+window) and [i+window+gap, ...).

    gap = units BETWEEN compared windows (0 = adjacent, negative = overlap,
    e.g. gap=-window/2 compares half-overlapping windows). step = how far the
    start advances per comparison (default window+gap: pairs tile the doc
    without re-measuring). Empty if the doc is too short.

    unit="tokens": whitespace-split words. unit="bytes": raw UTF-8 bytes
    (window < 255 keeps CK2 on its sparse8 fast path).
    """
    if window < 1 or (step is not None and step < 1):
        raise ValueError("window and step must be >= 1")
    get, n = _make_getter(text, unit, window)
    if step is None:
        step = window + max(gap, 0)
    second = window + gap  # offset of the compared window; gap > -window required
    if second <= 0:
        raise ValueError("gap must be > -window")
    scores = []
    i = 0
    while i + second + window <= n:
        scores.append(_distance(get(i), get(i + second), metric))
        i += step
    return scores


def random_change_series(
    text: str,
    window: int = 200,
    samples: int = 32,
    min_gap: int = 100,
    metric: str = "ck2",
    unit: str = "bytes",
) -> list[float]:
    """Distances between random window pairs at long range (gap >= min_gap).

    Seeded by a content hash: deterministic per input (RULES #5).
    Long-range self-similarity probes where local smoothing breaks down.
    """
    if window < 1 or samples < 1 or min_gap < 0:
        raise ValueError("bad window/samples/min_gap")
    get, n = _make_getter(text, unit, window)
    if n < 2 * window + min_gap:
        return []
    rng = random.Random(
        int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    )
    scores = []
    for _ in range(samples):
        i = rng.randrange(0, n - window + 1)
        regions = []  # valid j starts at least min_gap away from i's window
        if i - window - min_gap >= 0:
            regions.append((0, i - window - min_gap))
        if i + window + min_gap <= n - window:
            regions.append((i + window + min_gap, n - window))
        if not regions:
            regions.append((0, n - window))  # short doc: gap relaxes, j != i still holds
        lo, hi = regions[rng.randrange(len(regions))]
        j = rng.randrange(lo, hi + 1)
        if j == i:
            j = lo if lo != i else hi
        scores.append(_distance(get(i), get(j), metric))
    return scores


def burst_features(
    text: str,
    window: int = 20,
    gap: int = 0,
    step: int | None = None,
    metric: str = "ck2",
    unit: str = "tokens",
    mode: str = "step",
    samples: int = 32,
    min_gap: int = 100,
) -> dict[str, float]:
    """Summary stats of the change series. NaN stats when the doc is too short.

    mode="step": window pairs at i / i+window+gap. mode="random": random
    long-range pairs (window/samples/min_gap).
    """
    if mode == "step":
        series = change_series(text, window, gap, step, metric, unit)
    elif mode == "random":
        series = random_change_series(text, window, samples, min_gap, metric, unit)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    n = len(series)
    if n == 0:
        return {
            "count": 0.0,
            "mean": math.nan,
            "stdev": math.nan,
            "min": math.nan,
            "max": math.nan,
            "median": math.nan,
            "iqr": math.nan,
            "frac_near_identical": math.nan,
        }
    q1, med, q3 = statistics.quantiles(series, n=4)
    return {
        "count": float(n),
        "mean": statistics.fmean(series),
        "stdev": statistics.pstdev(series),
        "min": min(series),
        "max": max(series),
        "median": med,
        "iqr": q3 - q1,
        "frac_near_identical": sum(1 for s in series if s < 0.05) / n,
    }
