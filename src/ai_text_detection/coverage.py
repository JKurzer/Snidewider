"""Reference n-gram coverage features (fleet F2, promoted to the panel).

feature = fraction of a doc's word q-grams present in a pooled reference
counter per class (human/AI). Membership, not distance — the mega-profile
distance was composted (fleet B); coverage contrast survives.

LEAVE-ONE-OUT IS NOT OPTIONAL (fleet F3 v1 crater): any row whose source sits
inside the reference must be scored with its source-mates' grams subtracted,
else train/serve skew poisons the model. Use build_exclusions per source.

Pure functions; references are plain Counters (RULES #5).
"""

from __future__ import annotations

import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z0-9']+")
QS = (2, 3, 5)

COVERAGE_FEATURE_NAMES = tuple(
    f"cov{q}_{s}" for q in QS for s in ("hu", "ai", "contrast")
)


def doc_ngrams(text: str, q: int) -> set[tuple[str, ...]]:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    return {tuple(toks[i : i + q]) for i in range(len(toks) - q + 1)}


def build_reference(texts, q: int) -> Counter:
    """Pooled gram counts for a class. Deterministic (counts, then frozenset-like use)."""
    ref: Counter = Counter()
    for t in texts:
        ref.update(doc_ngrams(str(t), q))
    return ref


def coverage_features(
    text: str,
    ref_hu: dict[int, Counter],
    ref_ai: dict[int, Counter],
    exclude: dict[int, Counter] | None = None,
) -> dict[str, float]:
    """The 9 coverage features. `exclude` (per-q Counter) removes the doc's
    own source-mates from the reference: hits count only when
    ref_count > exclude_count. Pass None for rows outside the reference."""
    out: dict[str, float] = {}
    for q in QS:
        mine = doc_ngrams(text, q)
        if not mine:
            for s in ("hu", "ai", "contrast"):
                out[f"cov{q}_{s}"] = float("nan")
            continue
        vals = []
        for ref in (ref_hu[q], ref_ai[q]):
            if exclude is not None:
                ex = exclude[q]
                hits = sum(1 for g in mine if ref.get(g, 0) > ex.get(g, 0))
            else:
                hits = sum(1 for g in mine if g in ref)
            vals.append(hits / len(mine))
        out[f"cov{q}_hu"] = vals[0]
        out[f"cov{q}_ai"] = vals[1]
        out[f"cov{q}_contrast"] = vals[1] - vals[0]
    return out


def source_exclusion(texts, q: int) -> Counter:
    """Union of gram counts across a source's rows — the LOO exclusion term."""
    ex: Counter = Counter()
    for t in texts:
        ex.update(doc_ngrams(str(t), q))
    return ex
