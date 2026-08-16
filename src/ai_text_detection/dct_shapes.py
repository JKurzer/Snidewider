"""Tail-shaped DCT features — the D3 fleet's winning set (nobase, 50 feats).

Stock dct.dct_features describe the bulk of the segment-similarity
distribution; the low-FPR tail lives in its SHAPE. This set, measured on a
clean A/B/C protocol (docs/exp_d3.md): 4-det stack TPR@1e-3 0.340 vs 0.175
control, AUROC 0.934 vs 0.903.

Families (all from the K=5 segment encoding):
  paircos  seeded random non-adjacent pair-cosine quantiles + adj/nonadj gap
  bands    per-coefficient energy spectrum ||c_k|| stats + ratios to ||c_0||
  normpct  percentile profile of the segment-norm distribution
  drift    first-third vs last-third contrast (doc-level DCT-space drift)
  acosq    adjacent-cosine quantiles (tail of local smoothness)

Pair sampling is seeded by content hash — pure functions (RULES #5).
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from ai_text_detection.dct import dct_coefficients, embed_sentence

K_BANDS = 5
N_PAIRS = 16
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_Q_NAMES = ("p05", "p10", "p25", "p50", "p75", "p90", "p95", "iqr", "spread")


class _Doc:
    """Per-doc shared quantities: segment DCT vectors + band norms."""

    def __init__(self, text: str) -> None:
        self.vectors: list[np.ndarray] = []
        self.band_norms: list[list[float]] = []
        segments = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 2]
        for segment in segments:
            embedded = embed_sentence(segment)
            if embedded.shape[0] < 2:
                continue
            coeffs = dct_coefficients(embedded, K_BANDS)
            self.vectors.append(coeffs.reshape(-1))
            self.band_norms.append([float(np.linalg.norm(coeffs[k])) for k in range(K_BANDS)])
        self.n = len(self.vectors)
        self.adjacent = [self._cos(i, i + 1) for i in range(self.n - 1)]
        seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "little")
        rng = np.random.RandomState(seed)
        candidates = [(i, j) for i in range(self.n) for j in range(i + 2, self.n)]
        if len(candidates) > N_PAIRS:
            picks = rng.choice(len(candidates), size=N_PAIRS, replace=False)
            candidates = [candidates[p] for p in picks]
        self.pair_cos = [self._cos(i, j) for i, j in candidates]

    def _cos(self, i: int, j: int) -> float:
        a, b = self.vectors[i], self.vectors[j]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denom) if denom else 0.0

    def norms(self) -> np.ndarray:
        return np.array([float(np.linalg.norm(v)) for v in self.vectors])


def _quantiles(values) -> list[float]:
    if len(values) == 0:
        return [math.nan] * (len(QUANTILES) + 2)
    qs = list(np.quantile(np.asarray(values, dtype=float), QUANTILES))
    return qs + [qs[4] - qs[2], qs[5] - qs[1]]  # IQR, p90-p10


def _stats(values) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    return (float(np.mean(values)), float(np.std(values)))


def _paircos(doc: _Doc, out: dict[str, float]) -> None:
    for name, val in zip(_Q_NAMES, _quantiles(doc.pair_cos)):
        out[f"paircos_{name}"] = val
    if doc.pair_cos and doc.adjacent:
        out["paircos_gap"] = float(np.mean(doc.adjacent) - np.mean(doc.pair_cos))
    else:
        out["paircos_gap"] = math.nan


def _bands(doc: _Doc, out: dict[str, float]) -> None:
    if doc.n < 2:
        for k in range(K_BANDS):
            out[f"bands_c{k}_mean"] = out[f"bands_c{k}_std"] = math.nan
        for k in range(1, K_BANDS):
            out[f"bands_r{k}_mean"] = out[f"bands_r{k}_std"] = math.nan
        return
    bands = np.array(doc.band_norms)
    for k in range(K_BANDS):
        out[f"bands_c{k}_mean"], out[f"bands_c{k}_std"] = _stats(list(bands[:, k]))
    base = np.where(bands[:, 0] > 0, bands[:, 0], np.nan)
    for k in range(1, K_BANDS):
        out[f"bands_r{k}_mean"], out[f"bands_r{k}_std"] = _stats(list(bands[:, k] / base))


def _normpct(doc: _Doc, out: dict[str, float]) -> None:
    for name, val in zip(_Q_NAMES, _quantiles(doc.norms())):
        out[f"normpct_{name}"] = val


def _drift(doc: _Doc, out: dict[str, float]) -> None:
    if doc.n < 3:
        for key in ("cos", "adj", "ratio", "norm"):
            out[f"drift_{key}"] = math.nan
        return
    third = max(1, doc.n // 3)
    first_v = np.mean(doc.vectors[:third], axis=0)
    last_v = np.mean(doc.vectors[-third:], axis=0)
    denom = float(np.linalg.norm(first_v) * np.linalg.norm(last_v))
    ratios = [b[1] / b[0] if b[0] else 0.0 for b in doc.band_norms]
    norms = doc.norms()
    adj_first = doc.adjacent[: max(1, third - 1)]
    adj_last = doc.adjacent[-max(1, third - 1) :]
    out["drift_cos"] = float(first_v @ last_v / denom) if denom else 0.0
    out["drift_adj"] = float(np.mean(adj_first) - np.mean(adj_last))
    out["drift_ratio"] = float(np.mean(ratios[:third]) - np.mean(ratios[-third:]))
    out["drift_norm"] = float(np.median(norms[:third]) - np.median(norms[-third:]))


def _acosq(doc: _Doc, out: dict[str, float]) -> None:
    for name, val in zip(_Q_NAMES, _quantiles(doc.adjacent)):
        out[f"acosq_{name}"] = val


_SHAPE_FNS = (_paircos, _bands, _normpct, _drift, _acosq)


def dct_tail_features(text: str) -> dict[str, float]:
    """The 50-feature nobase set. NaN-filled for unusable docs."""
    doc = _Doc(text)
    out: dict[str, float] = {}
    for fn in _SHAPE_FNS:
        fn(doc, out)
    return out


def dct_tail_vector(text: str) -> list[float]:
    """Ordered feature vector (order = sorted feature names, stable)."""
    feats = dct_tail_features(text)
    return [feats[k] for k in sorted(feats)]
