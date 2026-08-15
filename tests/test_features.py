"""Feature extractor tests: profile stats sanity + vector shape + purity."""

import math

from ai_text_detection import features


def test_profile_stats_repetitive_vs_varied():
    repetitive = features.qgram_profile_features(" ".join(["same"] * 200))
    varied = features.qgram_profile_features(" ".join(f"w{i}" for i in range(200)))
    assert repetitive["qgram_distinct_ratio"] < varied["qgram_distinct_ratio"]
    assert repetitive["qgram_repeat_frac"] > varied["qgram_repeat_frac"]
    assert repetitive["qgram_max_share"] > varied["qgram_max_share"]


def test_vector_shape_and_names():
    text = " ".join(f"token{i % 53}" for i in range(400))
    vec = features.feature_vector(text)
    assert len(vec) == len(features.FEATURE_NAMES)
    assert all(v == v for v in vec)  # long doc: no NaNs


def test_short_doc_nan_behavior():
    vec = features.feature_vector("tiny doc")
    assert any(math.isnan(v) for v in vec)  # burst features can't compute
    profile_feats = features.qgram_profile_features("ab")
    assert math.isnan(profile_feats["qgram_distinct_ratio"])  # len < q


def test_pure_function():
    text = "some representative document " * 30
    assert features.feature_vector(text) == features.feature_vector(text)
