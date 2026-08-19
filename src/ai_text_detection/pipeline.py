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
from ai_text_detection.bigrams import BIGRAM_FEATURE_NAMES, bigram_rates
from ai_text_detection.bwt_stats import BWT_FEATURE_NAMES, bwt_features
from ai_text_detection.chargrams import CHARGRAM_FEATURE_NAMES, chargram_features
from ai_text_detection.charstat import CHARSTAT_FEATURE_NAMES, charstat_features
from ai_text_detection.collapse import COLLAPSE_FEATURE_NAMES, collapse_features
from ai_text_detection.coverage import COVERAGE_FEATURE_NAMES, coverage_features
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.exemplar import ExemplarBank, exemplar_features
from ai_text_detection.feature_sets import qgram12_vector, relative_vector
from ai_text_detection.shape import SHAPE_FEATURE_NAMES, shape_features
from ai_text_detection.stats_features import STAT_FEATURE_NAMES, WORD_RE, stat_features
from ai_text_detection.token_bigrams import REUSE_FEATURE_NAMES, token_reuse_features

BUNDLE = Path("data/derived/detector_bundle.pkl")


def featurize(text: str, artifacts: dict, *, csa_mode: str = "impute") -> np.ndarray:
    """The panel vector for one doc, in artifacts['feature_names'] order.

    csa_mode: "impute" (production default) fills the three csa_* columns
    with their A-fit imputation means -- the CSA trio measured ~zero marginal
    value in the mixture and costs ~27ms/doc (fleet_csa_ablation). "full"
    computes the real CSA measures (forensics/verification path).
    """
    tail = dct_tail_features(text)
    shape = shape_features(text)
    stats = stat_features(text)
    col = collapse_features(text)
    chr_ = charstat_features(text)  # frozen ENGLISH_CHAR_REF inside
    cov = coverage_features(text, artifacts["ref_hu"], artifacts["ref_ai"])
    b = text.encode("utf-8")
    n = max(1, len(b))
    names = artifacts["feature_names"]
    need_bwt = any(n_.startswith("bwt_") for n_ in names)
    # one CSA build serves both the csa trio (full mode) and the bwt block
    csa = _csa_native.csa_stats(b) if (csa_mode == "full" or need_bwt) else None
    if csa_mode == "full":
        csa_vals = [float(len(b)), csa["csa_wt_bytes"] / n, csa["csa_sada_bytes"] / n]
    else:
        means = artifacts["impute_means"]
        csa_vals = [float(means[names.index(f"csa_{k}")]) for k in ("n", "wt_rate", "sada_rate")]
    exf = exemplar_features(qgram.profile(b, 3),
                            artifacts["bank_ai"], artifacts["bank_hu"])
    feat_set = set(artifacts["feature_names"])
    ex_names = [k for k in artifacts["feature_names"]
                if k.startswith("ex_") and k != "ex_contrast_centroid"]
    row = (
        relative_vector(text)
        + qgram12_vector(text)
        + [exf[n] for n in ex_names]
        + [tail[k] for k in sorted(tail) if f"dct_{k}" in feat_set]
        + [shape[k] for k in SHAPE_FEATURE_NAMES]
        + [stats[k] for k in STAT_FEATURE_NAMES]
        + [cov[k] for k in COVERAGE_FEATURE_NAMES if k in feat_set]
        + [col[k] for k in COLLAPSE_FEATURE_NAMES if f"col_{k}" in feat_set]
        + [chr_[k] for k in CHARSTAT_FEATURE_NAMES if f"chr_{k}" in feat_set]
        + csa_vals
    )
    need_series = ("qg_s256_ck2_mean" in artifacts["feature_names"]
                   or any(n.startswith("s256_") for n in artifacts["feature_names"]))
    if need_series:
        from ai_text_detection import burst
        series = burst.random_change_series(text, window=150, samples=256,
                                            min_gap=50, metric="ck2", unit="tokens")
        if "qg_s256_ck2_mean" in artifacts["feature_names"]:
            row.append(float(np.mean(series)) if series else np.nan)
    if any(n.startswith("bg_") for n in artifacts["feature_names"]):
        rates = bigram_rates(text)
        row.extend(rates[k] for k in BIGRAM_FEATURE_NAMES if k in feat_set)
    if any(n.startswith("reuse_") for n in artifacts["feature_names"]):
        ru = token_reuse_features(text)
        row.extend(ru[k] for k in REUSE_FEATURE_NAMES)
    if any(k in feat_set for k in CHARGRAM_FEATURE_NAMES):
        cg = chargram_features(text)
        row.extend(cg[k] for k in CHARGRAM_FEATURE_NAMES if k in feat_set)
    if need_bwt:
        bw = bwt_features(text, bwt=None if csa is None else csa["bwt"])
        row.extend(bw[k] for k in BWT_FEATURE_NAMES)
    if "oct_hits" in artifacts["feature_names"]:
        from ai_text_detection.token_bigrams import oct_hits_features
        row.append(oct_hits_features(text)["oct_hits"])
    if "ex_contrast_centroid" in artifacts["feature_names"]:
        from ai_text_detection.exemplar import centroid_contrast
        row.append(centroid_contrast(qgram.profile(b, 3),
                                     artifacts["centroid_ai"],
                                     artifacts["centroid_hu"]))
    if any(n.startswith(("delta_", "wdelta_")) for n in artifacts["feature_names"]):
        row.extend(_delta_row(text))
    if any(n.startswith("cover_") or n == "wd_density" for n in artifacts["feature_names"]):
        from ai_text_detection.cover import COVER_FEATURE_NAMES, cover_features
        cv_ = cover_features(text)
        row.extend(cv_[k] for k in COVER_FEATURE_NAMES)
    if any(n.startswith("s256_") for n in artifacts["feature_names"]):
        if len(series) < 256:
            series = series + [np.nan] * (256 - len(series))
        row.extend(series)
    return np.array(row, dtype=float)


def _delta_row(text: str) -> list[float]:
    """The 13-feature delta family (distinct k-mer counts / k; word deltas).
    Kept local to the pipeline so the package stays free of fleet imports."""
    bts = text.encode("utf-8")
    out: list[float] = []
    ds: list[float] = []
    for k in range(1, 9):
        d = len(qgram.profile(bts, k)) / k if len(bts) >= k else np.nan
        out.append(d)
        ds.append(d)
    finite = [d for d in ds if np.isfinite(d)]
    if finite:
        out.extend([float(np.nanmax(ds)), float(np.nanargmax(ds) + 1)])
    else:
        out.extend([np.nan, np.nan])
    toks = [w.lower() for w in WORD_RE.findall(text)]
    for k in (1, 2, 3):
        grams = {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}
        out.append(len(grams) / k if toks else np.nan)
    return out


def featurize_batch(texts, artifacts: dict, *, csa_mode: str = "impute") -> np.ndarray:
    return np.array([featurize(str(t), artifacts, csa_mode=csa_mode) for t in texts])


def load_artifacts(path: Path = BUNDLE) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def impute(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    X = X.copy()
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(means, bad[1])
    return X
