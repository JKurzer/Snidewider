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

OCT_FEATURE_NAMES = ("oct_repeat_rate", "oct_repeat_abs")

OCT_HITS_FEATURE_NAMES = ("oct_hits", "oct_hits_rate")


def oct_hits_features(text: str, span: int = 8) -> dict[str, float]:
    """Donk's octgram recurrence: for each token, COUNT its appearances in
    the preceding octgram and the succeeding octgram ([oct] token [oct]),
    summed cumulatively over the doc. One doc-level score (plus its
    per-token rate). Every repeated pair contributes twice (seen from both
    sides); multiple occurrences in a window each count.
    """
    toks = [w.lower() for w in WORD_RE.findall(text)]
    n = len(toks)
    if n < 2 * span + 1:
        return {k: math.nan for k in OCT_HITS_FEATURE_NAMES}
    hits = 0
    for i, t in enumerate(toks):
        hits += toks[max(0, i - span):i].count(t)
        hits += toks[i + 1:i + span + 1].count(t)
    return {"oct_hits": float(hits), "oct_hits_rate": hits / n}


def oct_repeat_features(text: str, half: int = 4) -> dict[str, float]:
    """Adjacent-octgram recurrence: for each token, do any of its 8 nearest
    neighbors (+/-4 positions) equal it? Rate = fraction of tokens that
    recur inside their adjacent octgram. Local lexical recurrence -
    distinct from bigram/trigram surface repetition (which frontier models'
    anti-repeat glue suppresses).
    """
    toks = [w.lower() for w in WORD_RE.findall(text)]
    n = len(toks)
    if n < 2 * half + 2:
        return {k: math.nan for k in OCT_FEATURE_NAMES}
    hits = 0
    for i, t in enumerate(toks):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if t in toks[lo:i] or t in toks[i + 1:hi]:
            hits += 1
    return {"oct_repeat_rate": hits / n, "oct_repeat_abs": float(hits)}


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
