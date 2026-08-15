"""DCT sentence encoder (Almarwani & Diab 2021) + DCT-space document features.

Per the paper: a sentence is an N x d matrix of word embeddings; apply DCT-II
along the sentence axis per embedding dimension, keep the first K coefficients,
concatenate -> K*d vector. c[0] is proportional to the mean (averaging is the
K=1 special case); higher K captures word-order structure.

Document features (our addition, for the detector panel): adjacent-sentence
cosine similarity stats in DCT space (structural burstiness) plus the
order-energy ratio ||c[1]|| / ||c[0]||.

Embeddings: GloVe 6B 300d (data/derived/embeddings.npz via
scripts/build_embeddings.py). Lazy-loaded once, read-only (RULES #5: caching
fine, hidden state not).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

K = 2  # paper's best setting (c[0:1])
EMBEDDINGS = Path("data/derived/embeddings.npz")

_VOCAB: dict[str, int] | None = None
_MATRIX: np.ndarray | None = None

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _load() -> tuple[dict[str, int], np.ndarray]:
    global _VOCAB, _MATRIX
    if _MATRIX is None:
        data = np.load(EMBEDDINGS)
        _MATRIX = data["matrix"]
        _VOCAB = {w: i for i, w in enumerate(data["vocab"])}
    return _VOCAB, _MATRIX


def embed_sentence(sentence: str) -> np.ndarray:
    """N x d embedding matrix for a sentence; OOV tokens are skipped."""
    vocab, matrix = _load()
    indices = [vocab[t] for t in _TOKEN_RE.findall(sentence.lower()) if t in vocab]
    if not indices:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32)
    return matrix[indices]


def dct_coefficients(vectors: np.ndarray, k: int = K) -> np.ndarray:
    """First k DCT-II coefficients along axis 0. (N, d) -> (k, d).

    c[K] = sqrt(2/N) * sum_n v_n cos(pi/N (n + 1/2) K)  — paper Eq. verbatim.
    """
    n = vectors.shape[0]
    if n == 0:
        return np.zeros((k, vectors.shape[1]), dtype=np.float32)
    idx = np.arange(n)
    basis = math.sqrt(2.0 / n) * np.cos(np.pi / n * (idx + 0.5)[:, None] * np.arange(k)[None, :])
    return (basis.T @ vectors).astype(np.float32)


def sentence_vector(sentence: str, k: int = K) -> np.ndarray:
    """Flattened K*d DCT vector for a sentence."""
    return dct_coefficients(embed_sentence(sentence), k).reshape(-1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


DCT_FEATURE_NAMES = (
    "dct_adjacent_mean",
    "dct_adjacent_stdev",
    "dct_order_energy_mean",
    "dct_order_energy_stdev",
)


def dct_features(
    text: str,
    k: int = K,
    unit: str = "sentences",
    window: int = 24,
) -> dict[str, float]:
    """Document-level DCT features.

    unit="sentences": split on [.!?] (paper-style sentence encoder).
    unit="windows":   split into non-overlapping token windows of `window`
    words (uniform sampling instead of sentence boundaries).
    k = number of DCT coefficients kept (the "width").
    NaNs for docs with fewer than two usable segments.
    """
    if unit == "sentences":
        segments = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 2]
    elif unit == "windows":
        tokens = text.split()
        segments = [
            " ".join(tokens[i : i + window])
            for i in range(0, len(tokens) - window + 1, window)
        ]
    else:
        raise ValueError(f"unknown unit: {unit!r}")
    vectors = []
    energies = []
    for segment in segments:
        embedded = embed_sentence(segment)
        if embedded.shape[0] < 2:
            continue
        coeffs = dct_coefficients(embedded, k)
        vectors.append(coeffs.reshape(-1))
        base = float(np.linalg.norm(coeffs[0]))
        energies.append(float(np.linalg.norm(coeffs[1]) / base) if base and k > 1 else 0.0)
    if len(vectors) < 2:
        return {name: math.nan for name in DCT_FEATURE_NAMES}
    adjacent = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    return {
        "dct_adjacent_mean": float(np.mean(adjacent)),
        "dct_adjacent_stdev": float(np.std(adjacent)),
        "dct_order_energy_mean": float(np.mean(energies)),
        "dct_order_energy_stdev": float(np.std(energies)),
    }
