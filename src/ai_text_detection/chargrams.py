"""Char-trigram rates + CV-skeleton features (raw, per-doc, single pass).

Extends the bigram vector one order up: top-32 char trigrams of English
(list frozen from bucket-A humans, 2026-08-17; scripts/exp/build_trigram_list.py),
plus the consonant/vowel skeleton bigram rates (phonotactic texture) and the
word-initial char histogram entropy (alliteration-adjacent).

Pure functions (RULES #5).
"""

from __future__ import annotations

import math
import re
from collections import Counter

TRIGRAM_LIST = [' th', 'the', 'he ', 'nd ', ' an', 'and', 'ing', ' to', 'to ',
                ' in', 'ng ', 'ed ', ' of', 'er ', 'of ', 'in ', ' a ', 'is ',
                'on ', ' co', 'es ', 'e t', 's a', 'at ', 'ion', 're ', 'as ',
                ' be', 'e a', ' re', 'or ', 'n t']

TRIGRAM_FEATURE_NAMES = tuple(f"tg3_{t}" for t in TRIGRAM_LIST)

CV_FEATURE_NAMES = ("cv_cc_rate", "cv_cv_rate", "cv_vc_rate", "cv_vv_rate",
                    "initial_char_entropy")

_VOWELS = set("aeiou")
_WORD_RE = re.compile(r"[A-Za-z]+")


def _ent(counts, n: int) -> float:
    if n <= 0:
        return math.nan
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def chargram_features(text: str) -> dict[str, float]:
    s = text.lower()
    n3 = max(1, len(s) - 2)
    counts3 = Counter(zip(s, s[1:], s[2:]))
    out = {f"tg3_{t}": counts3.get(tuple(t), 0) / n3 for t in TRIGRAM_LIST}

    words = _WORD_RE.findall(s)
    letters = [c for c in s if c.isalpha()]
    cv = "".join("v" if c in _VOWELS else "c" for c in letters)
    cv_counts = Counter(zip(cv, cv[1:]))
    n_cv = max(1, len(cv) - 1)
    out["cv_cc_rate"] = cv_counts.get(("c", "c"), 0) / n_cv
    out["cv_cv_rate"] = cv_counts.get(("c", "v"), 0) / n_cv
    out["cv_vc_rate"] = cv_counts.get(("v", "c"), 0) / n_cv
    out["cv_vv_rate"] = cv_counts.get(("v", "v"), 0) / n_cv

    initials = Counter(w[0] for w in words if w)
    out["initial_char_entropy"] = _ent(initials, sum(initials.values()))
    return out


CHARGRAM_FEATURE_NAMES = TRIGRAM_FEATURE_NAMES + CV_FEATURE_NAMES
