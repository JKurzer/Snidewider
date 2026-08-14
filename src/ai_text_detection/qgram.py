"""q-gram distance/similarity — Ukkonen 1992, native-backed.

d_q(x,y) = L1 distance between q-gram count profiles (all substrings of
length q). Lower bound on edit distance: d_q(x,y) <= 2q * d_edit(x,y).
Bag distance (Bartolini et al. 2002) is the q=1 case with max(P, N)
instead of P + N.

Raw byte semantics: no case folding, no whitespace stripping. Feed UTF-8
bytes and do any normalization in the feature layer. There is deliberately
NO pure-Python fallback here — the shipped implementation is the native
port (scripts/build_native.py); tests carry a naive oracle to verify it.

Profile caching: `profile()` results can be reused via `distance_profiles`
/ `bag_distance_profiles` — profiles are plain lists of (int, int) tuples,
trivially serializable for corpus-scale work.
"""

from __future__ import annotations

from typing import Union

BytesLike = Union[bytes, bytearray, memoryview]

try:
    from ai_text_detection import _qgram_native
except ImportError as exc:  # pragma: no cover - depends on build state
    raise RuntimeError(
        "_qgram_native is not built. Run: .venv\\Scripts\\python scripts/build_native.py"
    ) from exc

Profile = list[tuple[int, int]]


def _as_bytes(s: BytesLike) -> bytes:
    return s if isinstance(s, bytes) else bytes(s)


def profile(a: BytesLike, q: int = 3) -> Profile:
    """Sorted (code, count) q-gram profile. Cacheable; reuse via *_profiles."""
    return _qgram_native.profile(_as_bytes(a), q)


def distance(a: BytesLike, b: BytesLike, q: int = 3) -> int:
    """Ukkonen q-gram distance: L1 between profiles. 0 = identical profiles."""
    return _qgram_native.distance(_as_bytes(a), _as_bytes(b), q)


def similarity(a: BytesLike, b: BytesLike, q: int = 3) -> float:
    """1 - d_q / max_d in [0, 1], where max_d = total q-grams on both sides.

    Degenerate when both strings are shorter than q (max_d = 0): returns 1.0
    (identical empty profiles). Callers should require len >= q.
    """
    A, B = _as_bytes(a), _as_bytes(b)
    max_d = max(0, len(A) - q + 1) + max(0, len(B) - q + 1)
    if max_d == 0:
        return 1.0
    return 1.0 - _qgram_native.distance(A, B, q) / max_d


def distance_profiles(x: Profile, y: Profile) -> int:
    """Ukkonen distance between cached profiles (must share q)."""
    pos, neg = _qgram_native.diff_profiles(x, y)
    return pos + neg


def bag_distance(a: BytesLike, b: BytesLike) -> int:
    """Bag distance (Bartolini 2002): max(P, N) over character multisets."""
    pos, neg = _qgram_native.diff(_as_bytes(a), _as_bytes(b), 1)
    return max(pos, neg)


def bag_similarity(a: BytesLike, b: BytesLike) -> float:
    """1 - bag / max(len) in [0, 1]. Empty/empty is defined as 1.0."""
    A, B = _as_bytes(a), _as_bytes(b)
    hi = max(len(A), len(B))
    if hi == 0:
        return 1.0
    return 1.0 - bag_distance(A, B) / hi


def bag_distance_profiles(x: Profile, y: Profile) -> int:
    """Bag distance between cached q=1 profiles."""
    pos, neg = _qgram_native.diff_profiles(x, y)
    return max(pos, neg)
