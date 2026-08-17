"""Comma-context / situational bigram features (Jin–Zaitsu lineage).

Punctuation CONTEXT, not punctuation rates (charstat owns the rates).
The tells: Oxford-comma habit, comma-splice tendency, which connectives
follow commas, quote-period ordering, sentence-end variety, and the
comma-per-sentence rhythm.

Pure functions (RULES #5); single pass per doc.
"""

from __future__ import annotations

import math
import re

COMMA_FEATURE_NAMES = (
    "oxford_rate", "splice_rate", "comma_and_rate", "comma_but_rate",
    "comma_because_rate", "comma_then_quote", "period_in_quotes",
    "comma_then_capital", "comma_then_lower", "end_variety",
    "sent_comma_mean", "sent_comma_stdev",
)

_SENT_RE = re.compile(r"[.!?]+")


def _rate(a: int, b: int) -> float:
    return a / b if b else math.nan


def comma_features(text: str) -> dict[str, float]:
    n_commas = text.count(",")
    if len(text) < 200 or n_commas < 3:
        return {k: math.nan for k in COMMA_FEATURE_NAMES}

    oxc = text.count(", and") + text.count(", or") + text.count(", but")
    plain = text.count(" and") + text.count(" or") + text.count(" but")

    lower = len(re.findall(r",\s+[a-z]", text))
    capital = len(re.findall(r",\s+[A-Z]", text))
    letter_ctx = lower + capital

    in_q = text.count('."')
    out_q = text.count('".')

    sents = [s for s in _SENT_RE.split(text) if s.strip()]
    comma_per_sent = [s.count(",") for s in sents]
    ends = set(m.group(0)[-1] for m in _SENT_RE.finditer(text))

    return {
        "oxford_rate": _rate(oxc, oxc + plain),
        "splice_rate": _rate(lower, letter_ctx),
        "comma_and_rate": _rate(text.count(", and"), n_commas),
        "comma_but_rate": _rate(text.count(", but"), n_commas),
        "comma_because_rate": _rate(text.count(", because"), n_commas),
        "comma_then_quote": _rate(text.count(',"'), n_commas),
        "period_in_quotes": _rate(in_q, in_q + out_q),
        "comma_then_capital": _rate(capital, letter_ctx),
        "comma_then_lower": _rate(lower, letter_ctx),
        "end_variety": _rate(len(ends), len(_SENT_RE.findall(text)) or 1),
        "sent_comma_mean": float(sum(comma_per_sent) / len(comma_per_sent)) if sents else math.nan,
        "sent_comma_stdev": float(
            (sum((c - sum(comma_per_sent) / len(comma_per_sent)) ** 2
                 for c in comma_per_sent) / len(comma_per_sent)) ** 0.5) if sents else math.nan,
    }
