"""Burst feature tests: pure-function invariants + separation sanity."""

import math
import random

from ai_text_detection import burst

RNG = random.Random(20260815)


def test_repetitive_doc_has_zero_change():
    text = " ".join(["lorem"] * 500)
    features = burst.burst_features(text)
    assert features["mean"] == 0.0
    assert features["stdev"] == 0.0
    assert features["frac_near_identical"] == 1.0


def test_varied_doc_has_high_change():
    # Diverse vocabulary: random 8-letter words share almost no char structure.
    words = [
        "".join(chr(RNG.randrange(97, 123)) for _ in range(8)) for _ in range(500)
    ]
    features = burst.burst_features(" ".join(words))
    assert features["mean"] > 0.1  # conservative floor; measured ~0.245
    assert features["frac_near_identical"] == 0.0


def test_repetitive_scores_below_varied():
    repetitive = burst.burst_features(" ".join(["same"] * 500))["mean"]
    varied = burst.burst_features(" ".join(f"w{i}" for i in range(500)))["mean"]
    assert repetitive < varied


def test_short_doc_returns_nan_stats():
    features = burst.burst_features("too short doc", window=20, gap=10)
    assert features["count"] == 0.0
    assert math.isnan(features["mean"])


def test_bag_metric_bounds_and_repetition():
    text = " ".join(f"word{i % 37}" for i in range(300))
    bag_series = burst.change_series(text, metric="bag")
    assert len(bag_series) == len(burst.change_series(text, metric="ck2"))
    assert all(0.0 <= s <= 1.0 for s in bag_series)
    repetitive = burst.burst_features(" ".join(["same"] * 500), metric="bag")
    assert repetitive["mean"] == 0.0


def test_deterministic_and_qgram_metric_works():
    text = " ".join(f"word{i % 37}" for i in range(300))
    assert burst.change_series(text, metric="ck2") == burst.change_series(text, metric="ck2")
    qgram_series = burst.change_series(text, metric="qgram")
    assert len(qgram_series) == len(burst.change_series(text, metric="ck2"))
    assert all(0.0 <= s <= 1.0 for s in qgram_series)
