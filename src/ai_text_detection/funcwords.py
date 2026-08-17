"""Closed-class function-word rates (the Mosteller-Wallace workhorse).

The oldest trick in forensic stylometry: function-word usage is involuntary
and stable per author. Rates per token over a frozen closed-class inventory,
plus category subrates including the discourse-marker set that LLMs lean on
("Moreover," / "Furthermore," / "It's important to note") and hedges.

Pure functions (RULES #5); word lists are frozen module constants.
"""

from __future__ import annotations

import math

from ai_text_detection.stats_features import WORD_RE

ARTICLES = frozenset("the a an".split())
PREPOSITIONS = frozenset(
    "of in to for with on at from by about into over after between through during "
    "under against among within without upon towards toward across behind beyond per via".split())
PRONOUNS = frozenset(
    "i me my mine we us our ours you your yours he him his she her hers it its they "
    "them their theirs this that these those who whom whose which what anyone someone "
    "everyone anybody somebody everybody nobody anything something everything nothing "
    "one ones oneself".split())
FIRST_PERSON = frozenset("i me my mine we us our ours".split())
SECOND_PERSON = frozenset("you your yours".split())
AUXILIARY = frozenset(
    "am is are was were be been being have has had do does did will would shall should "
    "can could may might must ought need dare".split())
CONJ_COORD = frozenset("and but or nor for so yet".split())
CONJ_SUBORD = frozenset(
    "because although though while if unless since as until when where whereas whether "
    "before once than".split())
DISCOURSE = frozenset(
    "however therefore moreover furthermore nevertheless nonetheless thus hence "
    "consequently accordingly additionally specifically importantly notably indeed "
    "overall finally firstly secondly thirdly meanwhile otherwise instead regardless".split())
HEDGE = frozenset(
    "perhaps maybe possibly probably likely unlikely somewhat rather quite fairly "
    "generally usually often sometimes rarely never always almost nearly approximately".split())
INTENSIFIER = frozenset(
    "very really extremely incredibly remarkably particularly especially highly "
    "totally completely absolutely entirely".split())

ALL_FW = (ARTICLES | PREPOSITIONS | PRONOUNS | AUXILIARY | CONJ_COORD | CONJ_SUBORD
          | DISCOURSE | HEDGE | INTENSIFIER)

FUNCWORD_FEATURE_NAMES = (
    "fw_total", "fw_article", "fw_preposition", "fw_pronoun", "fw_first_person",
    "fw_second_person", "fw_auxiliary", "fw_conj_coord", "fw_conj_subord",
    "fw_discourse", "fw_hedge", "fw_intensifier", "fw_distinct",
)


def _rate(tokens: list[str], bag: frozenset) -> float:
    return sum(1 for t in tokens if t in bag) / max(1, len(tokens))


def funcword_features(text: str) -> dict[str, float]:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    if len(toks) < 10:
        return {k: math.nan for k in FUNCWORD_FEATURE_NAMES}
    used = {t for t in toks if t in ALL_FW}
    return {
        "fw_total": _rate(toks, ALL_FW),
        "fw_article": _rate(toks, ARTICLES),
        "fw_preposition": _rate(toks, PREPOSITIONS),
        "fw_pronoun": _rate(toks, PRONOUNS),
        "fw_first_person": _rate(toks, FIRST_PERSON),
        "fw_second_person": _rate(toks, SECOND_PERSON),
        "fw_auxiliary": _rate(toks, AUXILIARY),
        "fw_conj_coord": _rate(toks, CONJ_COORD),
        "fw_conj_subord": _rate(toks, CONJ_SUBORD),
        "fw_discourse": _rate(toks, DISCOURSE),
        "fw_hedge": _rate(toks, HEDGE),
        "fw_intensifier": _rate(toks, INTENSIFIER),
        "fw_distinct": len(used) / len(ALL_FW),
    }
