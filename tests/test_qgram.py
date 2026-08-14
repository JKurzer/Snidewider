"""q-gram tests: native port vs naive oracle + published test vectors.

The naive dict implementation below is the obviously-correct oracle (lives
in tests only, never shipped). Published vectors come from dexyk/stringosim's
own test suite (whitespace-free cases only, so raw Ukkonen semantics agree).
"""

import random

import pytest

from ai_text_detection import qgram

RNG = random.Random(20260814)


# --- the oracle: dict-of-counts L1, plainly correct, plainly slow -----------
def oracle_profile(s: bytes, q: int) -> dict[bytes, int]:
    prof: dict[bytes, int] = {}
    for i in range(len(s) - q + 1):
        gram = s[i : i + q]
        prof[gram] = prof.get(gram, 0) + 1
    return prof


def oracle_diff(a: bytes, b: bytes, q: int) -> tuple[int, int]:
    p1, p2 = oracle_profile(a, q), oracle_profile(b, q)
    pos = neg = 0
    for g in p1.keys() | p2.keys():
        d = p1.get(g, 0) - p2.get(g, 0)
        if d > 0:
            pos += d
        else:
            neg -= d
    return pos, neg


def oracle_distance(a: bytes, b: bytes, q: int) -> int:
    pos, neg = oracle_diff(a, b, q)
    return pos + neg


def oracle_bag(a: bytes, b: bytes) -> int:
    return max(oracle_diff(a, b, 1))


# --- published vectors: stringosim q-gram_test.go (q=2, case-sensitive) -----
GO_VECTORS = [
    (b"", b"", 0),
    (b"xxxyyy", b"xxxyyy", 0),
    (b"xxxyyy", b"yyyxxx", 2),
    (b"xxyzxyyzzy", b"xyyxzyzxyzyx", 6),
    (b"xxyyzz", b"xxxzzz", 6),
    (b"asdlkajsdlkasdkj", b"fkdsjlkdf", 21),
    (b"STRING", b"sting", 9),
]


@pytest.mark.parametrize("a,b,expected", GO_VECTORS)
def test_published_vectors(a, b, expected):
    assert qgram.distance(a, b, q=2) == expected


def test_hand_computed():
    # q=3: all grams disjoint -> distance = total grams on both sides
    assert qgram.distance(b"abcde", b"abfde", q=3) == 6
    assert qgram.similarity(b"abcde", b"abfde", q=3) == 0.0
    # bag: aab {a:2,b:1} vs abb {a:1,b:2} -> P=1, N=1
    assert qgram.bag_distance(b"aab", b"abb") == 1
    assert qgram.distance(b"aab", b"abb", q=1) == 2  # L1 = P+N at q=1
    # len < q edge: "ab" has 0 trigrams, "abc" has 1
    assert qgram.distance(b"ab", b"abc", q=3) == 1
    assert qgram.similarity(b"ab", b"abc", q=3) == 0.0


def test_identity_and_bounds():
    assert qgram.distance(b"hello world", b"hello world", q=3) == 0
    assert qgram.similarity(b"hello world", b"hello world", q=3) == 1.0
    assert qgram.similarity(b"", b"", q=3) == 1.0
    assert qgram.bag_similarity(b"", b"") == 1.0
    with pytest.raises(ValueError):
        qgram.distance(b"a", b"b", q=0)
    with pytest.raises(ValueError):
        qgram.distance(b"a", b"b", q=9)


def rand_bytes(maxlen: int, alphabet: int) -> bytes:
    return bytes(RNG.randrange(alphabet) for _ in range(RNG.randrange(maxlen)))


def test_native_matches_oracle_fuzz():
    for q in range(1, 9):
        for alphabet in (2, 26, 256):
            for _ in range(15):
                a, b = rand_bytes(600, alphabet), rand_bytes(600, alphabet)
                assert qgram.distance(a, b, q) == oracle_distance(a, b, q), (q, a, b)
    for _ in range(50):
        a, b = rand_bytes(600, 26), rand_bytes(600, 26)
        assert qgram.bag_distance(a, b) == oracle_bag(a, b)


def test_profile_caching_round_trip():
    for _ in range(30):
        a, b = rand_bytes(300, 26), rand_bytes(300, 26)
        pa, pb = qgram.profile(a, 3), qgram.profile(b, 3)
        assert qgram.distance_profiles(pa, pb) == qgram.distance(a, b, 3)
        p1a, p1b = qgram.profile(a, 1), qgram.profile(b, 1)
        assert qgram.bag_distance_profiles(p1a, p1b) == qgram.bag_distance(a, b)
