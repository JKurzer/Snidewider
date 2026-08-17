"""FLEET E2 — fresh featurize vs cached columns, all families (dev A sample).

The 0.714-vs-0.7113 exam/ladder gap was never root-caused. If cached feature
columns are bit-identical to a fresh featurize of the same rows, the gap was
model-level (HGB binning on different-yet-equivalent data is deterministic,
so it would implicate sampling); if not, we have a cache bug. Also: feature
purity (same doc twice -> identical) and per-bucket NaN-rate audit.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_e2_cache_truth.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import qgram
from ai_text_detection.coverage import (
    COVERAGE_FEATURE_NAMES,
    QS,
    build_reference,
    coverage_features,
    source_exclusion,
)
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import ExemplarBank, bank_self_indices, exemplar_vector
from ai_text_detection.feature_sets import qgram12_vector, relative_vector
from ai_text_detection.charstat import CHARSTAT_FEATURE_NAMES, charstat_features
from ai_text_detection.collapse import COLLAPSE_FEATURE_NAMES, collapse_features
from ai_text_detection.shape import SHAPE_FEATURE_NAMES, shape_features
from ai_text_detection.stats_features import STAT_FEATURE_NAMES, stat_features

N_BANK = 150
SAMPLE = 60
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def featurize_all(text: str, bank_ai: ExemplarBank, bank_hu: ExemplarBank,
                  ai_self=None, hu_self=None, refs=None, exclude=None) -> list[float]:
    tail = dct_tail_features(text)
    shape = shape_features(text)
    stats = stat_features(text)
    col = collapse_features(text)
    chr_ = charstat_features(text)  # frozen ENGLISH_CHAR_REF
    row = (
        relative_vector(text)
        + qgram12_vector(text)
        + exemplar_vector(qgram.profile(text.encode("utf-8"), 3), bank_ai, bank_hu,
                          ai_self, hu_self)
        + [tail[k] for k in sorted(tail)]
        + [shape[k] for k in SHAPE_FEATURE_NAMES]
        + [stats[k] for k in STAT_FEATURE_NAMES]
    )
    if refs is not None:
        cov = coverage_features(text, refs[0], refs[1], exclude=exclude)
        row += [cov[k] for k in COVERAGE_FEATURE_NAMES]
    row += [col[k] for k in COLLAPSE_FEATURE_NAMES]
    row += [chr_[k] for k in CHARSTAT_FEATURE_NAMES]
    return row


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    a = buckets["A"]
    bank_ai = ExemplarBank.from_texts([str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts([str(t) for t in a[a.model == "human"].generation[:N_BANK]])

    cache = np.load("data/derived/full_features.npz")
    Xa = cache["X_A"]

    ai_self, hu_self = bank_self_indices([str(m) for m in a.model], N_BANK)
    # A rows' coverage is vs B+C-built refs (cross-bucket, no LOU needed)
    bc = pd.concat([buckets["B"], buckets["C"]])
    refs = ({q: build_reference(bc[bc.model == "human"].generation, q) for q in QS},
            {q: build_reference(bc[bc.model != "human"].generation, q) for q in QS})
    ref_sums = {q: None for q in QS}  # coverage handled via refs above
    texts_a = [str(t) for t in a.generation]
    sources_a = list(a.source_id)

    print("== exemplar LOO: bank members must NOT self-match ==")
    names0 = list(cache["feature_names"])
    hu_min_col = names0.index("ex_hu_min")
    check("row 0 (bank_hu member) ex_hu_min > 0", Xa[0, hu_min_col] > 0,
          f"(got {Xa[0, hu_min_col]:.4f})")

    print("== fresh vs cache, 60 A rows (sampled across the bucket) ==")
    idx = np.linspace(0, len(a) - 1, SAMPLE).astype(int)
    gens = list(a.generation)
    mismatch = 0
    for i in idx:
        fresh = np.array(featurize_all(str(gens[i]), bank_ai, bank_hu,
                                       ai_self[i], hu_self[i], refs=refs,
                                       exclude=None), dtype=float)
        cached = Xa[i]
        if not np.allclose(np.nan_to_num(fresh), np.nan_to_num(cached), atol=1e-9):
            d = np.abs(np.nan_to_num(fresh) - np.nan_to_num(cached))
            mismatch += 1
            print(f"    row {i}: max |diff| {d.max():.3e} at col {int(d.argmax())}", flush=True)
    check(f"fresh==cache (60 rows x {Xa.shape[1]} feats)", mismatch == 0, f"({mismatch} mismatched rows)")

    print("== purity: same doc twice ==")
    twice = featurize_all(str(gens[idx[0]]), bank_ai, bank_hu, ai_self[idx[0]], hu_self[idx[0]])
    check("pure function", np.array_equal(np.nan_to_num(twice), np.nan_to_num(
        featurize_all(str(gens[idx[0]]), bank_ai, bank_hu, ai_self[idx[0]], hu_self[idx[0]]))))

    print("== NaN-rate audit per bucket ==")
    names = list(cache["feature_names"])
    for b in "ABC":
        X = cache[f"X_{b}"]
        rates = np.isnan(X).mean(axis=0)
        worst = rates.argsort()[-5:][::-1]
        top = ", ".join(f"{names[i]} {rates[i]:.2f}" for i in worst if rates[i] > 0)
        print(f"  {b}: {int((rates > 0).sum())} cols with NaN; worst: {top}", flush=True)


if __name__ == "__main__":
    main()
