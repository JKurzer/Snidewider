"""Relative-window features: full coverage (no NaN) where absolute windows drop."""

import math

from ai_text_detection.features import FEATURE_NAMES, document_features
from ai_text_detection.features_relative import (
    FEATURE_NAMES_RELATIVE,
    document_features_relative,
    midrange_params,
    short_range_window,
)

SHORT_DOC = "the quick brown fox jumps over the lazy dog and runs away " * 8
LONG_DOC = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do " * 200


def test_param_scaling():
    assert short_range_window(2048) == 64  # >= 512B keeps the 64B window
    assert short_range_window(200) == 25  # 200//8
    assert midrange_params(1200) == (150, 100)  # clamped at the top
    assert midrange_params(96) == (16, 8)  # clamped at the floor
    w, g = midrange_params(30)  # shrink-to-fit instead of NaN
    assert 2 * w + g <= 30


def test_no_qgram_total_no_nan_on_short_doc():
    feats = document_features_relative(SHORT_DOC)
    assert tuple(feats) == FEATURE_NAMES_RELATIVE
    assert "qgram_total" not in feats
    assert "qgram_total" in FEATURE_NAMES  # the absolute set still has the proxy
    assert not any(math.isnan(v) for v in feats.values())


def test_absolute_windows_drop_the_same_doc():
    assert math.isnan(document_features(SHORT_DOC)["midrange_mean"])


def test_degenerate_tiny_doc_still_finite():
    feats = document_features_relative("a b c d e f g h i j")
    assert not any(math.isnan(v) for v in feats.values())


def test_pure_and_deterministic():
    assert document_features_relative(LONG_DOC) == document_features_relative(LONG_DOC)
