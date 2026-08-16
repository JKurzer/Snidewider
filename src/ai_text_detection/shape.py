"""Shape mappings: text -> small-alphabet symbol streams for CK2.

CK2 doesn't need characters, just symbols from a small alphabet. Two maps:

  skeleton_map — every word condenses to '='; whitespace and punctuation stay.
                 CK2 over this is an edit distance over pure sentence
                 STRUCTURE (word-count rhythm per sentence, nothing lexical).

  dct_run_map  — per short token run (8 tokens), DCT-II of the token-length
                 sequence, quantized into one byte symbol (c0 hi-nibble,
                 c1 lo-nibble). CK2 over these symbols measures the visual
                 rhythm of the text — does it *look* normative.

Features per map = burst stats of the symbol stream (adjacent-window CK2
mean/stdev + seeded random long-range pair mean/stdev).
"""

from __future__ import annotations

import math
import re

import numpy as np

from ai_text_detection import burst

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
RUN = 8  # tokens per DCT run
MAXLEN = 15  # token lengths clip into [0..15] nibbles


def skeleton_map(text: str) -> bytes:
    """Words -> '=', whitespace/punct preserved."""
    return _WORD_RE.sub("=", text).encode("utf-8")


def dct_run_map(text: str) -> bytes:
    """One quantized DCT symbol per RUN-token run."""
    lengths = [min(len(w), MAXLEN) for w in _WORD_RE.findall(text)]
    symbols = bytearray()
    idx = np.arange(RUN)
    for i in range(0, len(lengths) - RUN + 1, RUN):
        v = np.array(lengths[i : i + RUN], dtype=float)
        basis = math.sqrt(2.0 / RUN) * np.cos(np.pi / RUN * (idx + 0.5)[:, None] * np.arange(2)[None, :])
        c = basis.T @ v  # c0 ~ mean length, c1 ~ length slope
        q0 = min(15, max(0, int(c[0] / math.sqrt(2.0 * RUN) * 2)))  # c0/sqrt(2N)=mean len; *2 spreads 0-7.5 -> nibble
        q1 = min(15, max(0, int(c[1] + 8)))  # slope roughly in [-8..8] -> nibble
        symbols.append((q0 << 4) | q1)
    return bytes(symbols)


SHAPE_MAPS = ("skeleton", "dct_run")

SHAPE_FEATURE_NAMES = tuple(
    f"{m}_{stat}" for m in SHAPE_MAPS for stat in ("step_mean", "step_stdev", "rand_mean", "rand_stdev")
)


def shape_features(text: str) -> dict[str, float]:
    """CK2 burst stats over each mapped symbol stream. NaN if too short."""
    out: dict[str, float] = {}
    for name, mapper in (("skeleton", skeleton_map), ("dct_run", dct_run_map)):
        stream = mapper(text)
        stepped = burst.burst_features(stream.decode("latin-1"), window=32, gap=0, unit="bytes")
        rnd = burst.burst_features(
            stream.decode("latin-1"), window=32, samples=32, min_gap=32, unit="bytes", mode="random"
        )
        out[f"{name}_step_mean"] = stepped["mean"]
        out[f"{name}_step_stdev"] = stepped["stdev"]
        out[f"{name}_rand_mean"] = rnd["mean"]
        out[f"{name}_rand_stdev"] = rnd["stdev"]
    return out
