"""CK2 similarity — Python bindings with a pure-Python reference port.

CK2 (Donk's name; the headers call it CKS / ck::sparse) is a linear-time
approximation of Levenshtein/Damerau-Levenshtein distance. Score semantics:
0.0 = identical, 1.0 = maximally different — despite the historical
"similarity" name in the C++ sources, it behaves as a normalized distance.

Usage prefers the native extension (`_ck2_native`, built by
scripts/build_ck2.py from the vendored headers in native/). The pure-Python
port below is a faithful transcription of ck::sparse and exists as the
correctness oracle in tests and as a last-resort fallback. It is O(m+n) but
with pure-Python constants — do not use it at scale.

Provenance: ck_similarity_sparse{,8}.hpp, novel implementation of the lapsed
CKS patent (see StatisticalMeasuresReferences/Readme.md).
"""

from __future__ import annotations

from typing import Union

BytesLike = Union[bytes, bytearray, memoryview]

try:  # native fast path (built in-place by scripts/build_ck2.py)
    from ai_text_detection import _ck2_native

    _HAS_NATIVE = True
except ImportError:  # pragma: no cover - depends on build state
    _ck2_native = None
    _HAS_NATIVE = False


def _as_bytes(s: BytesLike) -> bytes:
    if isinstance(s, bytes):
        return s
    return bytes(s)


def _positions(s: bytes) -> dict[int, list[int]]:
    pos: dict[int, list[int]] = {}
    for i, c in enumerate(s):
        pos.setdefault(c, []).append(i)
    return pos


def _match_one(xspan: list[int], yspan: list[int]) -> tuple[float, float, float]:
    """Greedy nearest-neighbour matching of one character class.

    Returns (D_add, sa_add, sb_add) — mirrors ck::sparse::matchOne exactly.
    """
    swapped = False
    if len(xspan) > len(yspan):
        xspan, yspan = yspan, xspan
        swapped = True

    i = j = 0
    long_size = len(yspan)
    stack: list[int] = []
    d_add = 0.0

    while i < len(xspan):
        xhead = xspan[i]
        i += 1
        while j < long_size and yspan[j] < xhead:
            stack.append(yspan[j])
            j += 1

        s_empty = not stack
        qy_empty = j >= long_size

        if not s_empty and not qy_empty:
            left = xhead - stack[-1]
            right = yspan[j] - xhead
            if left < right:
                d_add += left
                stack.pop()
            else:
                d_add += right
                j += 1
        elif not s_empty:
            d_add += xhead - stack[-1]
            stack.pop()
        elif not qy_empty:
            d_add += yspan[j] - xhead
            j += 1

    leftover = float(len(stack) + (long_size - j))
    return d_add, (leftover if swapped else 0.0), (0.0 if swapped else leftover)


def measures(a: BytesLike, b: BytesLike) -> dict[str, float]:
    """Intermediate CK measures (Sa, Sb, D, S, n) plus score. Pure-Python oracle."""
    A, B = _as_bytes(a), _as_bytes(b)
    if len(A) != len(B):
        X, Y = (A, B) if len(A) < len(B) else (B, A)
    else:
        X, Y = (A, B) if A <= B else (B, A)

    xpos, ypos = _positions(X), _positions(Y)
    sa = sb = d = 0.0

    for c, xspan in xpos.items():
        yspan = ypos.get(c)
        if yspan is not None:
            d_add, sa_add, sb_add = _match_one(xspan, yspan)
            d += d_add
            sa += sa_add
            sb += sb_add
        else:
            sa += len(xspan)
    for c, yspan in ypos.items():
        if c not in xpos:
            sb += len(yspan)

    n = len(Y)
    s = max(sa, sb)
    if n == 0:
        score = 0.0
    else:
        percent_ordered = (n * n - d) / (n * n)
        percent_similar_chars = (n - s) / n
        score = 1.0 - percent_ordered * percent_similar_chars
    return {"Sa": sa, "Sb": sb, "D": d, "S": s, "n": float(n), "score": score}


def similarity(a: BytesLike, b: BytesLike) -> float:
    """CK2 score in [0, 1]: 0.0 = identical, 1.0 = maximally different."""
    if _HAS_NATIVE:
        return float(_ck2_native.similarity(_as_bytes(a), _as_bytes(b)))
    return measures(a, b)["score"]


HAS_NATIVE = _HAS_NATIVE
