"""BWT run-structure features (the Wheeler edge sequence of the text).

A text's de Bruijn graph IS a Wheeler graph, and the BWT of the text is its
edge-label sequence under the Wheeler order (Gagie/Manzini/Sirén 2017). So
the run statistics below ARE Wheeler-graph statistics, computed from the
real BWT exposed by the vendored sdsl harness (_csa_native).

The benched condensate pack had bwt_runs_rate (count only); these are the
run-length DISTRIBUTION stats plus the BWT char histogram entropy.

Pure functions (RULES #5).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ai_text_detection import _csa_native

BWT_FEATURE_NAMES = ("bwt_run_max", "bwt_run_mean", "bwt_run_p90",
                     "bwt_run_entropy", "bwt_char_entropy")


def _ent(counts, n: int) -> float:
    if n <= 0:
        return math.nan
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def bwt_features(text: str) -> dict[str, float]:
    b = text.encode("utf-8")
    if len(b) < 200:
        return {k: math.nan for k in BWT_FEATURE_NAMES}
    bwt = np.asarray(_csa_native.csa_stats(b)["bwt"])
    # run lengths of consecutive equal BWT symbols
    cuts = np.nonzero(np.diff(bwt) != 0)[0] + 1
    runs = np.diff(np.concatenate(([0], cuts, [len(bwt)])))
    run_counts = Counter(int(r) for r in runs)
    return {
        "bwt_run_max": float(runs.max()),
        "bwt_run_mean": float(runs.mean()),
        "bwt_run_p90": float(np.percentile(runs, 90)),
        "bwt_run_entropy": _ent(run_counts, len(runs)),
        "bwt_char_entropy": _ent(Counter(bwt.tolist()), len(bwt)),
    }
