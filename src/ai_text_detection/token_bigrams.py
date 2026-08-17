"""Token (word) bigram rate features (raw, per-doc, single pass).

Per-doc rates of the top-64 word bigrams of English prose (list frozen from
bucket-A humans, 2026-08-17; see scripts/exp/build_token_bigram_list.py).
Feature = count(bigram) / total word bigrams in the doc.

Pure functions (RULES #5).
"""

from __future__ import annotations

import math
from collections import Counter

from ai_text_detection.stats_features import WORD_RE

TOKEN_BIGRAM_LIST = [
    ("of", "the"), ("in", "the"), ("to", "the"), ("and", "the"), ("in", "a"),
    ("on", "the"), ("for", "the"), ("with", "the"), ("to", "be"), ("at", "the"),
    ("from", "the"), ("is", "a"), ("1", "2"), ("of", "a"), ("it", "is"),
    ("as", "a"), ("with", "a"), ("by", "the"), ("to", "a"), ("that", "the"),
    ("as", "the"), ("is", "the"), ("br", "br"), ("one", "of"), ("he", "is"),
    ("on", "a"), ("it", "was"), ("over", "the"), ("1", "4"), ("and", "i"),
    ("will", "be"), ("he", "was"), ("into", "the"), ("for", "a"), ("and", "a"),
    ("of", "his"), ("the", "first"), ("i", "was"), ("have", "been"),
    ("they", "are"), ("i", "have"), ("add", "the"), ("has", "been"),
    ("that", "he"), ("the", "same"), ("i", "am"), ("was", "a"), ("this", "is"),
    ("into", "a"), ("can", "be"), ("to", "make"), ("all", "the"), ("by", "a"),
    ("out", "of"), ("to", "get"), ("as", "well"), ("1", "cup"),
    ("2", "tablespoons"), ("the", "world"), ("in", "this"), ("is", "not"),
    ("the", "story"), ("2", "cup"), ("such", "as"),
]

TOKEN_BIGRAM_FEATURE_NAMES = tuple(f"tg_{a}_{b}" for a, b in TOKEN_BIGRAM_LIST)


def token_bigram_rates(text: str) -> dict[str, float]:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    n = max(1, len(toks) - 1)
    counts = Counter(zip(toks, toks[1:]))
    return {f"tg_{a}_{b}": counts.get((a, b), 0) / n for a, b in TOKEN_BIGRAM_LIST}


REUSE_FEATURE_NAMES = ("peak_reuse", "peak_reuse_abs", "reuse_ge2", "reuse_ge3",
                       "top_share")


def token_reuse_features(text: str) -> dict[str, float]:
    """Peak bigram reuse: how hard the doc leans on its favorite bigram.

    Short human prose rarely repeats one bigram much; machines lean.
    """
    toks = [w.lower() for w in WORD_RE.findall(text)]
    n = len(toks) - 1
    if n < 20:
        return {k: math.nan for k in REUSE_FEATURE_NAMES}
    counts = Counter(zip(toks, toks[1:]))
    top_type, top_count = counts.most_common(1)[0]
    n_types = len(counts)
    return {
        "peak_reuse": top_count / n,
        "peak_reuse_abs": float(top_count),
        "reuse_ge2": sum(1 for c in counts.values() if c >= 2) / n_types,
        "reuse_ge3": sum(1 for c in counts.values() if c >= 3) / n_types,
        "top_share": top_count / sum(counts.values()),
    }
