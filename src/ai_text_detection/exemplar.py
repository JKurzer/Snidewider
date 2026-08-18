"""Exemplar-proximity features: doc-vs-CORPUS q-gram distances.

Structural complement to features.py's SELF-similarity families.
Hypothesis (F4): AI text resembles other AI text (a handful of generator
models); human text is more diverse. So distance to an AI exemplar bank —
and the contrast against a human bank — should separate the classes.

Banks are cached q-gram profiles (qgram.profile) built ONCE from TRAIN-half
docs; per-doc work is qgram.distance_profiles against the cached profiles.
Leave-one-out: a doc that sits in a bank passes its index so it never
matches itself (keeps train/test feature distributions comparable).

Pure functions / read-only banks (RULES #5); caching lives in ExemplarBank.
"""

from __future__ import annotations

import statistics

import numpy as np
from dataclasses import dataclass

from ai_text_detection import qgram

try:
    from ai_text_detection import _qgram_native as _qn
except ImportError:  # pragma: no cover
    _qn = None

Q = 3

EXEMPLAR_FEATURE_NAMES = (
    "ex_ai_min",
    "ex_ai_mean",
    "ex_ai_p10",
    "ex_hu_min",
    "ex_hu_mean",
    "ex_hu_p10",
    "ex_ai_mean_raw",
    "ex_hu_mean_raw",
    "ex_contrast_min",
    "ex_contrast_mean",
    "ex_contrast_p10",
    "ex_contrast_mode",
)


def profile_total(profile: qgram.Profile) -> int:
    """Total q-gram occurrences behind a profile (== doc bytes - q + 1)."""
    return sum(c for _, c in profile)


def bank_self_indices(models: list[str], n_bank: int) -> tuple[list[int | None], list[int | None]]:
    """Per-row (ai_self_index, hu_self_index) for banks built from the first
    n_bank AI/human rows of the same frame, in frame order.

    NOT optional for any row inside a bank: without it the row matches
    itself (distance 0) and every min/p10 stat is skewed — train/serve skew
    (fleet F3 crater; cache builders pre-2026-08-16 shipped this bug).
    """
    ai_seen = hu_seen = 0
    ai_idx: list[int | None] = []
    hu_idx: list[int | None] = []
    for m in models:
        if m == "human":
            hu_idx.append(hu_seen if hu_seen < n_bank else None)
            ai_idx.append(None)
            hu_seen += 1
        else:
            ai_idx.append(ai_seen if ai_seen < n_bank else None)
            hu_idx.append(None)
            ai_seen += 1
    return ai_idx, hu_idx


@dataclass
class ExemplarBank:
    """Cached exemplar profiles + precomputed totals.

    Totals are computed once at build: summing counts per distance call
    would make scoring O(bank x profile size) instead of O(bank).
    """

    profiles: list[qgram.Profile]
    totals: list[int]

    @classmethod
    def from_texts(cls, texts: list[str], q: int = Q) -> "ExemplarBank":
        profiles = [qgram.profile(t.encode("utf-8"), q) for t in texts]
        return cls(profiles, [profile_total(p) for p in profiles])

    def __len__(self) -> int:
        return len(self.profiles)


def _distances(
    doc_profile: qgram.Profile,
    doc_total: int,
    bank: ExemplarBank,
    self_index: int | None,
) -> tuple[list[float], list[float]]:
    """(raw, normalized) Ukkonen distances to the bank, leave-one-out aware.

    normalized = raw / (total_doc + total_ex) in [0, 1] — exactly
    1 - qgram.similarity(doc, exemplar). Raw distances grow with doc
    length; normalized ones don't (length confound removed).
    """
    raw, norm = [], []
    for i, (ex, ex_total) in enumerate(zip(bank.profiles, bank.totals)):
        if i == self_index:
            continue
        d = qgram.distance_profiles(doc_profile, ex)
        raw.append(float(d))
        denom = doc_total + ex_total
        norm.append(d / denom if denom else 0.0)
    return raw, norm


def _p10(values: list[float]) -> float:
    """10th percentile. Inclusive method: stays inside [min, max], so tiny
    banks (tests) don't extrapolate below 0; at 200 values it's within a
    hair of the exclusive method anyway."""
    return statistics.quantiles(values, n=10, method="inclusive")[0]


def exemplar_features(
    doc_profile: qgram.Profile,
    ai_bank: ExemplarBank,
    hu_bank: ExemplarBank,
    ai_self_index: int | None = None,
    hu_self_index: int | None = None,
) -> dict[str, float]:
    """The 11 exemplar-proximity features for one doc profile.

    min/mean/p10 normalized distance to each bank, raw mean distance to
    each bank (length-confounded control), and AI-minus-human contrasts
    (negative = closer to the AI bank = AI-like).
    """
    doc_total = profile_total(doc_profile)
    if _qn is not None:  # native fast path: banks resident in C++ (perf fleet)
        key = f"{len(ai_bank)}:{len(hu_bank)}:{hash(tuple(ai_bank.totals))}"
        if _qn.banks_key() != key:
            _qn.load_banks(ai_bank.profiles, ai_bank.totals,
                           hu_bank.profiles, hu_bank.totals, key)
        d = _qn.bank_distances(doc_profile, doc_total,
                               -1 if ai_self_index is None else ai_self_index,
                               -1 if hu_self_index is None else hu_self_index)
        ai_raw = [v for v in d["ai_raw"].tolist() if v == v]
        ai_n = [v for v in d["ai_norm"].tolist() if v == v]
        hu_raw = [v for v in d["hu_raw"].tolist() if v == v]
        hu_n = [v for v in d["hu_norm"].tolist() if v == v]
    else:
        ai_raw, ai_n = _distances(doc_profile, doc_total, ai_bank, ai_self_index)
        hu_raw, hu_n = _distances(doc_profile, doc_total, hu_bank, hu_self_index)
    ai_min, ai_mean, ai_p10 = min(ai_n), statistics.fmean(ai_n), _p10(ai_n)
    hu_min, hu_mean, hu_p10 = min(hu_n), statistics.fmean(hu_n), _p10(hu_n)

    def _mode(vals: list[float]) -> float:
        # densest neighborhood of the bank: center of the fullest of 20
        # equal-width bins over the normalized-distance range
        hist, edges = np.histogram(vals, bins=20)
        mid = int(np.argmax(hist))
        return float((edges[mid] + edges[mid + 1]) / 2)

    return {
        "ex_ai_min": ai_min,
        "ex_ai_mean": ai_mean,
        "ex_ai_p10": ai_p10,
        "ex_hu_min": hu_min,
        "ex_hu_mean": hu_mean,
        "ex_hu_p10": hu_p10,
        "ex_ai_mean_raw": statistics.fmean(ai_raw),
        "ex_hu_mean_raw": statistics.fmean(hu_raw),
        "ex_contrast_min": ai_min - hu_min,
        "ex_contrast_mean": ai_mean - hu_mean,
        "ex_contrast_p10": ai_p10 - hu_p10,
        "ex_contrast_mode": _mode(ai_n) - _mode(hu_n),
    }


def exemplar_vector(
    doc_profile: qgram.Profile,
    ai_bank: ExemplarBank,
    hu_bank: ExemplarBank,
    ai_self_index: int | None = None,
    hu_self_index: int | None = None,
) -> list[float]:
    """Ordered feature vector (order = EXEMPLAR_FEATURE_NAMES)."""
    feats = exemplar_features(doc_profile, ai_bank, hu_bank, ai_self_index, hu_self_index)
    return [feats[name] for name in EXEMPLAR_FEATURE_NAMES]
