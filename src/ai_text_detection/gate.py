"""gate.py — the no-FP tripwire, consolidated from the G1-G3 fleet hunt.

A union of zero-FPR stats over dct_run symbol streams: each component fires
when its stat lands strictly above the frozen bar (calibrated at the A+B
human maximum, so dev FPR is 0 by construction; honest FPR comes from C).
Fires = "suspiciously uniform rhythm stretch" — a small slice of AI docs
auto-flagged with ~zero false positives; the panel handles everything else.

Components and thresholds are a frozen spec (data/derived/gate_spec.json,
built by scripts/build_gate.py). Pure per doc (RULES #5): streams and series
are deterministic functions of the text.

Bytes-native series helpers (burst.py's str path double-encodes high bytes;
the dct_run symbols are 0..255 and must stay single-byte for window math).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

import numpy as np

from ai_text_detection import ck2

SPEC_PATH = Path("data/derived/gate_spec.json")

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
MAXLEN = 15


def dct_run_map_short(text: str, run: int, levels: int) -> bytes:
    """dct_run_map with knobs: `run` tokens/symbol, `levels`/coefficient."""
    words = _WORD_RE.findall(text)
    n_runs = len(words) // run
    if n_runs == 0:
        return b""
    lengths = np.array(
        [min(len(w), MAXLEN) for w in words[: n_runs * run]], dtype=float
    ).reshape(n_runs, run)
    idx = np.arange(run)
    basis = math.sqrt(2.0 / run) * np.cos(np.pi / run * (idx + 0.5)[:, None] * np.arange(2)[None, :])
    c = lengths @ basis
    shift = levels.bit_length() - 1
    q0 = np.clip((c[:, 0] / math.sqrt(2.0 * run) * (levels / 8.0)).astype(int), 0, levels - 1)
    q1 = np.clip(((c[:, 1] + 8.0) * (levels / 16.0)).astype(int), 0, levels - 1)
    return ((q0 << shift) | q1).astype(np.uint8).tobytes()


def step_series(stream: bytes, window: int, gap: int = 0) -> list[float]:
    n = len(stream)
    second = window + gap
    out, i = [], 0
    while i + second + window <= n:
        out.append(ck2.similarity(stream[i : i + window], stream[i + second : i + second + window]))
        i += window + max(gap, 0)
    return out


def random_series(stream: bytes, window: int, samples: int, min_gap: int) -> list[float]:
    n = len(stream)
    if n < 2 * window + min_gap:
        return []
    rng = random.Random(int.from_bytes(hashlib.sha256(stream).digest()[:4], "big"))
    out = []
    for _ in range(samples):
        i = rng.randrange(0, n - window + 1)
        regions = []
        if i - window - min_gap >= 0:
            regions.append((0, i - window - min_gap))
        if i + window + min_gap <= n - window:
            regions.append((i + window + min_gap, n - window))
        if not regions:
            regions.append((0, n - window))
        lo, hi = regions[rng.randrange(len(regions))]
        j = rng.randrange(lo, hi + 1)
        if j == i:
            j = lo if lo != i else hi
        out.append(ck2.similarity(stream[i : i + window], stream[j : j + window]))
    return out


def series_stats(series: list[float]) -> dict[str, float]:
    n = len(series)
    if n == 0:
        return {k: math.nan for k in ("mean", "stdev", "min", "max", "frac")}
    return {
        "mean": statistics.fmean(series),
        "stdev": statistics.pstdev(series) if n >= 2 else math.nan,
        "min": min(series),
        "max": max(series),
        "frac": sum(1 for s in series if s < 0.05) / n,
    }


def component_score(text: str, comp: dict) -> float:
    """One gate component's score for a doc (may be NaN if the stream is short)."""
    stream = dct_run_map_short(text, comp["run"], comp["levels"])
    if comp["mode"] == "step":
        series = step_series(stream, comp["window"])
    else:
        series = random_series(stream, comp["window"], comp["samples"], comp["window"])
    val = series_stats(series)[comp["stat"]]
    return -val if comp["direction"] == "low" else val


class Gate:
    """The frozen union gate. flag() True = auto-flagged as AI (zero-FP tripwire)."""

    def __init__(self, spec: dict) -> None:
        self.components = spec["components"]
        self.meta = spec.get("meta", {})

    @classmethod
    def load(cls, path: Path = SPEC_PATH) -> "Gate":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def report(self, text: str) -> dict[str, float]:
        """Per-component fired status (for inspection/debug)."""
        out = {}
        for comp in self.components:
            score = component_score(text, comp)
            out[comp["name"]] = score >= comp["threshold"] if score == score else False
        return out

    def flag(self, text: str) -> bool:
        return any(self.report(text).values())
