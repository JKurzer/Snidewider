"""Character-distribution aberration features (single-pass, no comparisons).

Per-doc character census: category rates, punctuation spectrum, whitespace
structure, run lengths, entropies, and a chi-squared aberration score against
a reference English char distribution (frozen constant, gate-spec style —
derived from bucket-A humans; see scripts/exp/fleet_distill.py).

The classic AI tells live here: em-dashes, curly quotes, tidy spacing.

Pure functions (RULES #5); the reference distribution is an explicit
parameter, never hidden state.
"""

from __future__ import annotations

import math
import string
import unicodedata
from collections import Counter

import numpy as np

# Frozen from bucket-A humans (dev fold, raid train_none) on 2026-08-16 by
# fleet_distill.py's --freeze-ref; printable ASCII only, sums to 1.
ENGLISH_CHAR_REF: dict[str, float] = {' ': 0.16652375742881642, '!': 0.00026610685668667397, '"': 0.0015516986487685166, '#': 2.3653942816593243e-06, '$': 0.00021939031962390232, '%': 0.00012713994263918867, '&': 6.563969131604624e-05, "'": 0.0021974512876615122, '(': 0.0008302533928624228, ')': 0.0008379409242778155, '*': 0.00010348599982259543, '+': 9.461577126637297e-06, ',': 0.009472221400904764, '-': 0.002064397859318175, '.': 0.008819963927737205, '/': 0.000794181130067118, '0': 0.0013660151976582597, '1': 0.002043109310783241, '2': 0.0013240294491588067, '3': 0.0005931226161260755, '4': 0.000591348570414831, '5': 0.0005540936104786966, '6': 0.0003453475651222613, '7': 0.0002862127080807782, '8': 0.00038023713077673635, '9': 0.0005996274504006387, ':': 0.0003370686851364537, ';': 0.0005103338162679991, '<': 0.0003181455308831791, '=': 2.424529138700807e-05, '>': 0.0003175541823127643, '?': 0.0002726116909612371, '@': 1.1826971408296621e-06, '[': 1.774045711244493e-05, '\\': 8.042340557641701e-05, ']': 1.8923154253274594e-05, '^': 1.655775997161527e-05, '_': 2.779338280949706e-05, '`': 8.278879985807634e-06, 'a': 0.0649525442772242, 'b': 0.013331362171431951, 'c': 0.02406138198160906, 'd': 0.03004937760562964, 'e': 0.09555542414476213, 'f': 0.016862895833949322, 'g': 0.016845155376836878, 'h': 0.03945300257236628, 'i': 0.057843943112267523, 'j': 0.0014724579403329293, 'k': 0.006851364536826232, 'l': 0.034322462375447206, 'm': 0.021299192809201384, 'n': 0.05577422311581562, 'o': 0.0591307176014902, 'p': 0.016638183377191685, 'q': 0.0007291327873214866, 'r': 0.049271754235534135, 's': 0.052887850743620825, 't': 0.0695337216522279, 'u': 0.021868661482510866, 'v': 0.008590520682416251, 'w': 0.014330149906862601, 'x': 0.0016439490257532303, 'y': 0.013691493450814583, 'z': 0.000843263061411549, '{': 4.494249135152716e-05, '|': 5.91348570414831e-06, '}': 4.494249135152716e-05, '~': 1.774045711244493e-06}

CHARSTAT_BASE_NAMES = (
    "chi2_en", "char_entropy", "bigram_cond_entropy",
    "upper_rate", "space_rate", "nonascii_rate", "control_rate",
    "comma_rate", "period_rate", "semicolon_rate", "colon_rate",
    "dash_rate", "emdash_rate", "quote_straight_rate", "quote_curly_rate",
    "paren_rate", "ellipsis_rate",
    "double_space_rate", "newline_rate", "para_len_mean", "para_len_stdev",
    "max_char_run", "vowel_rate",
)
CHARSTAT_FEATURE_NAMES = CHARSTAT_BASE_NAMES


def _rate(text: str, chars: str) -> float:
    return sum(1 for c in text if c in chars) / max(1, len(text))


def charstat_features(text: str, ref: dict[str, float] | None = None) -> dict[str, float]:
    n = max(1, len(text))
    low = text.lower()

    # chi-squared vs reference printable-English distribution
    if ref is None:
        ref = ENGLISH_CHAR_REF
    if ref:
        counts = Counter(c for c in low if c in ref)
        total = sum(counts.values())
        chi2 = math.nan
        if total > 100:
            chi2 = sum((counts.get(c, 0) - total * p) ** 2 / (total * p)
                       for c, p in ref.items() if p > 0) / total
    else:
        chi2 = math.nan

    # entropies
    freq = Counter(low)
    char_h = -sum((c / n) * math.log2(c / n) for c in freq.values())
    joint = Counter(zip(low, low[1:]))
    first = Counter(low[:-1])
    m = max(1, len(low) - 1)
    cond_h = -sum((cnt / m) * math.log2(cnt / first[c1])
                  for (c1, _), cnt in joint.items())

    # categories
    cats = Counter(unicodedata.category(c) for c in text)
    control = sum(v for k, v in cats.items() if k.startswith("C")) / n

    # whitespace structure
    paras = [p for p in text.split("\n") if p.strip()]
    plen = [len(p.split()) for p in paras] or [0]

    # run lengths
    max_run = run = 1
    for i in range(1, len(text)):
        run = run + 1 if text[i] == text[i - 1] else 1
        max_run = max(max_run, run)

    return {
        "chi2_en": chi2,
        "char_entropy": char_h,
        "bigram_cond_entropy": cond_h,
        "upper_rate": _rate(text, string.ascii_uppercase),
        "space_rate": _rate(text, " \t"),
        "nonascii_rate": sum(1 for c in text if ord(c) > 127) / n,
        "control_rate": control,
        "comma_rate": _rate(low, ","),
        "period_rate": _rate(low, "."),
        "semicolon_rate": _rate(low, ";"),
        "colon_rate": _rate(low, ":"),
        "dash_rate": _rate(low, "-"),
        "emdash_rate": _rate(text, "—–"),
        "quote_straight_rate": _rate(text, "\"'"),
        "quote_curly_rate": _rate(text, "\u201c\u201d\u2018\u2019"),
        "paren_rate": _rate(low, "()"),
        "ellipsis_rate": text.count("...") / n + _rate(text, "\u2026"),
        "double_space_rate": text.count("  ") / n,
        "newline_rate": _rate(text, "\n"),
        "para_len_mean": float(np.mean(plen)),
        "para_len_stdev": float(np.std(plen)) if len(plen) > 1 else math.nan,
        "max_char_run": float(max_run),
        "vowel_rate": _rate(low, "aeiou"),
    }
