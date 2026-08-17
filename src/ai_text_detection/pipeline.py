"""pipeline.py — production featurize: raw text -> the panel vector.

Single entry point replicating the cache computation exactly, in cache
column order. Reference artifacts (exemplar banks, coverage refs) come from
a bundle built by scripts/build_detector_bundle.py; the char reference is
frozen in charstat.ENGLISH_CHAR_REF; the CSA measures come from _csa_native.

RULES #5: featurize is a pure function of (text, artifacts).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ai_text_detection import _csa_native, qgram
from ai_text_detection.charstat import CHARSTAT_FEATURE_NAMES, charstat_features
from ai_text_detection.collapse import COLLAPSE_FEATURE_NAMES, collapse_features
from ai_text_detection.coverage import COVERAGE_FEATURE_NAMES, coverage_features
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.exemplar import ExemplarBank, exemplar_vector
from ai_text_detection.feature_sets import qgram12_vector, relative_vector
from ai_text_detection.shape import SHAPE_FEATURE_NAMES, shape_features
from ai_text_detection.stats_features import STAT_FEATURE_NAMES, stat_features

BUNDLE = Path("data/derived/detector_bundle.pkl")


def featurize(text: str, artifacts: dict) -> np.ndarray:
    """The full panel vector for one doc, in artifacts['feature_names'] order."""
    tail = dct_tail_features(text)
    shape = shape_features(text)
    stats = stat_features(text)
    col = collapse_features(text)
    chr_ = charstat_features(text)  # frozen ENGLISH_CHAR_REF inside
    cov = coverage_features(text, artifacts["ref_hu"], artifacts["ref_ai"])
    b = text.encode("utf-8")
    csa = _csa_native.csa_stats(b)
    n = max(1, len(b))
    row = (
        relative_vector(text)
        + qgram12_vector(text)
        + exemplar_vector(qgram.profile(b, 3), artifacts["bank_ai"], artifacts["bank_hu"])
        + [tail[k] for k in sorted(tail)]
        + [shape[k] for k in SHAPE_FEATURE_NAMES]
        + [stats[k] for k in STAT_FEATURE_NAMES]
        + [cov[k] for k in COVERAGE_FEATURE_NAMES]
        + [col[k] for k in COLLAPSE_FEATURE_NAMES]
        + [chr_[k] for k in CHARSTAT_FEATURE_NAMES]
        + [float(len(b)), csa["csa_wt_bytes"] / n, csa["csa_sada_bytes"] / n]
    )
    if "qg_s256_ck2_mean" in artifacts["feature_names"]:
        from ai_text_detection import burst
        s = burst.random_change_series(text, window=150, samples=256, min_gap=50,
                                       metric="ck2", unit="tokens")
        row.append(float(np.mean(s)) if s else np.nan)
    return np.array(row, dtype=float)


def featurize_batch(texts, artifacts: dict) -> np.ndarray:
    return np.array([featurize(str(t), artifacts) for t in texts])


def load_artifacts(path: Path = BUNDLE) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def impute(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    X = X.copy()
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(means, bad[1])
    return X
