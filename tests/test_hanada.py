"""Hanada substring-search tests: native vs brute-force oracle.

The oracle answers the paper's problem statement (Sec. 2.2) directly: for
each start i, scan every j in scope(i), compute d_q naively, pick the min
(longest on ties), report iff <= k. O(|t| * k * (|p|+k)) — fine at test sizes.
"""

import random

import pytest

from ai_text_detection import qgram

RNG = random.Random(20260814)


def naive_profile(s: bytes, q: int) -> dict[bytes, int]:
    prof: dict[bytes, int] = {}
    for i in range(len(s) - q + 1):
        gram = s[i : i + q]
        prof[gram] = prof.get(gram, 0) + 1
    return prof


def naive_dq(a: bytes, b: bytes, q: int) -> int:
    pa, pb = naive_profile(a, q), naive_profile(b, q)
    return sum(abs(pa.get(g, 0) - pb.get(g, 0)) for g in pa.keys() | pb.keys())


def oracle_search(t: bytes, p: bytes, q: int, k: int) -> list[tuple[int, int]]:
    """Ground truth for the paper's problem (scope-restricted, longest-on-ties)."""
    n, m = len(t), len(p)
    if n < q or m < q or not (0 <= k <= m - q):
        return []
    out = []
    for i in range(n - q + 1):
        b, e = i + m - 1 - k, min(i + m - 1 + k, n - 1)
        if b > e:
            break
        best_d, best_j = None, None
        for j in range(b, e + 1):
            d = naive_dq(t[i : j + 1], p, q)
            if best_d is None or d < best_d or (d == best_d and j > best_j):
                best_d, best_j = d, j
        if best_d <= k:
            out.append((i, best_j + 1))
    return out


def check_both(t: bytes, p: bytes, q: int, k: int) -> None:
    expected = oracle_search(t, p, q, k)
    assert qgram.search(t, p, q=q, k=k) == expected, (t, p, q, k)
    assert qgram.search_ukkonen(t, p, q=q, k=k) == expected, (t, p, q, k)


def test_paper_example_cabaab():
    # Paper Sec. 2.2: t="cabaab", p="abab", k=2, q=2; at 1-based i=2 the min
    # distance 1 is achieved at j=4 and j=6 -> longest wins: t[2..6].
    hits = qgram.search(b"cabaab", b"abab", q=2, k=2)
    assert (1, 6) in hits
    assert hits == oracle_search(b"cabaab", b"abab", 2, 2)


def test_paper_example_table2():
    # Table 2 / Fig. 2 example (Array-Search walkthrough), now end-to-end.
    t, p = b"aaaccaaababc", b"aaabbc"
    check_both(t, p, 2, 3)


def test_exact_match_is_found():
    t = b"the quick brown fox jumps over the lazy dog"
    p = b"brown fox"
    hits = qgram.search(t, p, q=2, k=0)
    start = t.index(p)
    assert (start, start + len(p)) in hits


def test_k0_single_slot_scope_regression():
    # k=0 -> 1-position scope; the previous right edge falls outside it and
    # the update loops skip it. The edge rule must not read it stale.
    t = b"zzzz" + b"abcdefg" + b"yyyyyyyyyy"
    check_both(t, b"abcdefg", 3, 0)
    check_both(t, b"abcdefg", 5, 0)


def test_validation():
    with pytest.raises(ValueError):
        qgram.search(b"abc", b"ab", q=3, k=0)  # len(p) < q
    with pytest.raises(ValueError):
        qgram.search(b"abcabc", b"abcabc", q=3, k=4)  # k > len(p) - q
    assert qgram.search(b"ab", b"abcdef", q=3, k=1) == []  # len(t) < q


def rand_bytes(n: int, alphabet: int) -> bytes:
    return bytes(RNG.randrange(alphabet) for _ in range(n))


def test_fuzz_vs_oracle():
    for _ in range(25):
        alphabet = RNG.choice((2, 20, 26, 256))
        t = rand_bytes(RNG.randrange(30, 300), alphabet)
        m = RNG.randrange(4, 30)
        q = RNG.randrange(1, min(6, m) + 1)
        if RNG.random() < 0.5 and len(t) >= m:
            start = RNG.randrange(len(t) - m + 1)  # paper-style: p from t
            p = t[start : start + m]
        else:
            p = rand_bytes(m, alphabet)
        k = RNG.randrange(0, m - q + 1)
        check_both(t, p, q, k)


def test_native_self_consistency_large():
    # Both engines must agree on bigger inputs where the oracle is too slow.
    for _ in range(5):
        t = rand_bytes(5_000, 20)
        start = RNG.randrange(4_000)
        p = t[start : start + 100]
        assert qgram.search(t, p, q=5, k=60) == qgram.search_ukkonen(t, p, q=5, k=60)


def test_hits_are_within_k():
    t = rand_bytes(400, 26)
    p = t[100:130]
    for start, end in qgram.search(t, p, q=3, k=15):
        assert naive_dq(t[start:end], p, 3) <= 15
