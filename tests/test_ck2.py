"""CK2 tests: invariants + port-vs-native agreement.

The pure-Python port (ck2.measures) is the oracle; the native extension must
agree with it exactly (same algorithm, same evaluation order).
"""

import random

import pytest

from ai_text_detection import ck2

RNG = random.Random(20260814)


def rand_bytes(maxlen: int = 300, alphabet: int = 256) -> bytes:
    n = RNG.randrange(0, maxlen)
    lo, hi = (0, alphabet)
    return bytes(RNG.randrange(lo, hi) for _ in range(n))


CASES = [
    (b"", b""),
    (b"", b"abc"),
    (b"abc", b""),
    (b"aaaa", b"aaaa"),
    (b"ab", b"ba"),
    (b"aaa", b"bbb"),
    (b"kitten", b"sitting"),
    (b"the quick brown fox", b"the quick brown fox jumps"),
    (bytes(range(256)), bytes(range(255, -1, -1))),
    (b"a" * 254, b"a" * 255),  # sparse8 boundary
    (b"x" * 300, b"y" * 300),  # forces the >=255 fallback path
    (bytes([0, 255, 0, 255]), bytes([255, 0, 255, 0])),
] + [(rand_bytes(), rand_bytes()) for _ in range(60)]


def test_identical_is_zero():
    assert ck2.similarity(b"hello world", b"hello world") == 0.0
    assert ck2.similarity(b"", b"") == 0.0


def test_disjoint_is_one():
    assert ck2.similarity(b"aaa", b"bbb") == 1.0


def test_reversal_hand_computed():
    # Worked out from the C++ source: X=b"ab", Y=b"ba" -> D=2, S=0, n=2
    # score = 1 - ((4-2)/4) * ((2-0)/2) = 0.5
    assert ck2.similarity(b"ab", b"ba") == pytest.approx(0.5)


def test_symmetric():
    for _ in range(50):
        a, b = rand_bytes(80), rand_bytes(80)
        assert ck2.similarity(a, b) == ck2.similarity(b, a)


def test_score_bounds():
    for a, b in CASES:
        score = ck2.similarity(a, b)
        assert 0.0 <= score <= 1.0, (a, b, score)


def test_port_measures_known_case():
    m = ck2.measures(b"ab", b"ba")
    assert (m["Sa"], m["Sb"], m["D"], m["S"], m["n"]) == (0.0, 0.0, 2.0, 0.0, 2.0)


@pytest.mark.skipif(not ck2.HAS_NATIVE, reason="native extension not built")
def test_native_matches_port():
    for a, b in CASES:
        native = ck2._ck2_native.similarity(a, b)
        oracle = ck2.measures(a, b)["score"]
        assert native == pytest.approx(oracle, abs=1e-12), (a, b)


@pytest.mark.skipif(not ck2.HAS_NATIVE, reason="native extension not built")
def test_native_measures_match_port():
    for a, b in CASES[:20]:
        nm = ck2._ck2_native.measures(a, b)
        om = ck2.measures(a, b)
        for key in ("Sa", "Sb", "D", "S", "n"):
            assert nm[key] == pytest.approx(om[key], abs=1e-9), (key, a, b)
