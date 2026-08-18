"""Covering-number + within-doc local density (metric-space features).

A doc's char 8-gram windows embedded as 8-d byte points (Chebyshev metric,
r = 0.2*std, cKDTree). cover_balls: greedy ball-cover count - the raw
covering number (how many neighborhoods span this doc's own material; the
hidden-dimensionality read). wd_density: mean |ball| / N - within-doc local
density, no reference needed.

Pure functions (RULES #5); NaN when too short.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

COVER_FEATURE_NAMES = ("cover_balls", "cover_balls_rate", "wd_density")

_WIN = 8
_N_WIN = 512


def _windows(text: str) -> np.ndarray | None:
    b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)
    span = len(b) - _WIN + 1
    if span < 16:
        return None
    starts = np.linspace(0, span - 1, min(_N_WIN, span)).astype(int)
    return b[starts[:, None] + np.arange(_WIN)[None, :]]


def cover_features(text: str) -> dict[str, float]:
    X = _windows(text)
    if X is None:
        return {k: math.nan for k in COVER_FEATURE_NAMES}
    r = 0.2 * float(X.std())
    if r == 0:
        return {k: math.nan for k in COVER_FEATURE_NAMES}
    tree = cKDTree(X)
    balls = tree.query_ball_point(X, r, p=np.inf)
    n = len(X)
    sizes = np.array([len(ball) for ball in balls])
    density = float(sizes.mean() / n)

    uncovered = np.ones(n, dtype=bool)
    n_balls = 0
    for i in np.argsort(-sizes):
        if uncovered[i]:
            n_balls += 1
            uncovered[np.asarray(balls[i], dtype=int)] = False
    return {
        "cover_balls": float(n_balls),
        "cover_balls_rate": n_balls / n,
        "wd_density": density,
    }
