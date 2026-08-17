"""Char-bigram rate features (raw, per-doc, single pass).

The Ishihara-shop's System-1, generalized: per-doc rates of the top-64
char bigrams of English (list frozen from bucket-A humans, 2026-08-17;
see scripts/exp/build_bigram_list.py). Each feature = count(bigram) /
total bigrams in the doc. Raw feature family for the learners.

Pure functions (RULES #5).
"""

from __future__ import annotations

import math
from collections import Counter

BIGRAM_LIST = ['e ', ' t', 'th', ' a', 's ', 'he', 'in', 'd ', 't ', 'an', 'n ',
               'er', ' s', ' i', 're', ' o', 'on', ' w', 'r ', 'nd', ' c', ', ',
               ' b', 'y ', 'at', 'es', ' h', 'en', 'to', 'o ', 'or', 'te', ' m',
               ' f', 'ng', 'ar', 'ed', 'is', 'st', '. ', 'ti', 'it', 'al', ' p',
               've', 'ou', 'le', 'ha', 'as', 'se', 'nt', 'ea', 'g ', 'f ', ' d',
               'me', 'of', 'hi', 'l ', 'a ', 'ro', 'co', 'ne', 'll']

BIGRAM_FEATURE_NAMES = tuple(f"bg_{a}{b}" for a, b in BIGRAM_LIST)


def bigram_rates(text: str) -> dict[str, float]:
    s = text.lower()
    n = max(1, len(s) - 1)
    counts = Counter(zip(s, s[1:]))
    out = {}
    for a, b in BIGRAM_LIST:
        out[f"bg_{a}{b}"] = counts.get((a, b), 0) / n
    return out
