"""exp_d3: DCT feature-shape search for the low-FPR tail.

The stock 4 DCT doc features (adjacent-cos mean/stdev + order-energy
mean/stdev) solo at TPR@1e-3 = 0.010 -- mean/stdev are bulk statistics and
the tail signal lives in the distribution shape. This script builds five
alternative shape families from the same DCT segment encoding (K=5 bands):

  base     reference 4 (control arm)
  paircos  seeded random NON-adjacent segment-pair cosines + local/global gap
  bands    per-coefficient energy stats: ||c_k|| mean/std, k=0..4, plus
           ||c_k||/||c_0|| ratio stats, k=1..4 (order-spectrum profile)
  normpct  percentiles of segment-norm distribution (p05..p95, IQR, p90-p10)
  drift    first-third vs last-third contrast in DCT space (doc-level drift)
  acosq    adjacent-cosine QUANTILES (p05..p95 + IQR) instead of mean/stdev

Protocol (RULES #2/#4): detectors train on A, shape selection on B, final
numbers on C exactly once. Stack = HGB(random_state=7) on B's 4 scores
(cached relative-burst/qgram12/exemplar + ours). Randomness is seeded per
doc (md5 of text) -- features stay pure functions (RULES #5).

Usage: see docs/exp_d3.md / mission briefing (PYTHONPATH = this worktree).
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.dct import dct_coefficients, embed_sentence
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

K_BANDS = 5  # DCT width for band/norm features (mission: K up to 4-6)
N_PAIRS = 16  # sampled non-adjacent pairs per doc
FEATURE_VERSION = 3  # bump when any shape definition changes
CACHE = Path(__file__).resolve().parent.parent / "data" / "derived" / "d3_features.npz"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# ---------------------------------------------------------------- featurizer
class _Doc:
    """Shared per-doc quantities: segment DCT vectors + derived stats."""

    def __init__(self, text: str) -> None:
        self.vectors: list[np.ndarray] = []
        self.band_norms: list[list[float]] = []  # per segment: ||c_0||..||c_{K-1}||
        segments = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 2]
        for segment in segments:
            embedded = embed_sentence(segment)
            if embedded.shape[0] < 2:
                continue
            coeffs = dct_coefficients(embedded, K_BANDS)
            self.vectors.append(coeffs.reshape(-1))
            self.band_norms.append([float(np.linalg.norm(coeffs[k])) for k in range(K_BANDS)])
        self.n = len(self.vectors)
        self.adjacent = [self._cos(i, i + 1) for i in range(self.n - 1)]
        # seeded non-adjacent pairs: pure function of the text (RULES #5)
        seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "little")
        rng = np.random.RandomState(seed)
        candidates = [(i, j) for i in range(self.n) for j in range(i + 2, self.n)]
        if len(candidates) > N_PAIRS:
            picks = rng.choice(len(candidates), size=N_PAIRS, replace=False)
            candidates = [candidates[p] for p in picks]
        self.pair_cos = [self._cos(i, j) for i, j in candidates]

    def _cos(self, i: int, j: int) -> float:
        a, b = self.vectors[i], self.vectors[j]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denom) if denom else 0.0

    def norms(self) -> np.ndarray:
        return np.array([float(np.linalg.norm(v)) for v in self.vectors])


def _quantiles(values: list[float] | np.ndarray) -> list[float]:
    if len(values) == 0:
        return [math.nan] * (len(QUANTILES) + 2)
    qs = list(np.quantile(np.asarray(values, dtype=float), QUANTILES))
    return qs + [qs[4] - qs[2], qs[5] - qs[1]]  # + IQR, p90-p10


def _stats(values: list[float]) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    return [float(np.mean(values)), float(np.std(values))]


def shape_base(doc: _Doc) -> dict[str, float]:
    """Reference 4 shapes (control)."""
    if doc.n < 2:
        return {f"base_{k}": math.nan for k in ("adjmean", "adjstd", "oemean", "oestd")}
    ratios = [b[1] / b[0] if b[0] else 0.0 for b in doc.band_norms]
    return {
        "base_adjmean": float(np.mean(doc.adjacent)),
        "base_adjstd": float(np.std(doc.adjacent)),
        "base_oemean": float(np.mean(ratios)),
        "base_oestd": float(np.std(ratios)),
    }


def shape_paircos(doc: _Doc) -> dict[str, float]:
    """Seeded non-adjacent pair cosines: global smoothness vs local."""
    out: dict[str, float] = {}
    qs = _quantiles(doc.pair_cos)
    for name, val in zip(("p05", "p10", "p25", "p50", "p75", "p90", "p95", "iqr", "spread"), qs):
        out[f"paircos_{name}"] = val
    if doc.pair_cos and doc.adjacent:
        out["paircos_gap"] = float(np.mean(doc.adjacent) - np.mean(doc.pair_cos))
    else:
        out["paircos_gap"] = math.nan
    return out


def shape_bands(doc: _Doc) -> dict[str, float]:
    """Per-coefficient energy bands: the order spectrum as a profile."""
    out: dict[str, float] = {}
    if doc.n < 2:
        for k in range(K_BANDS):
            out[f"bands_c{k}_mean"] = out[f"bands_c{k}_std"] = math.nan
        for k in range(1, K_BANDS):
            out[f"bands_r{k}_mean"] = out[f"bands_r{k}_std"] = math.nan
        return out
    bands = np.array(doc.band_norms)  # (n, K)
    for k in range(K_BANDS):
        out[f"bands_c{k}_mean"], out[f"bands_c{k}_std"] = _stats(list(bands[:, k]))
    base = np.where(bands[:, 0] > 0, bands[:, 0], np.nan)
    for k in range(1, K_BANDS):
        ratio = bands[:, k] / base
        out[f"bands_r{k}_mean"], out[f"bands_r{k}_std"] = _stats(list(ratio))
    return out


def shape_normpct(doc: _Doc) -> dict[str, float]:
    """Percentile profile of segment-norm distribution."""
    out: dict[str, float] = {}
    qs = _quantiles(doc.norms())
    for name, val in zip(("p05", "p10", "p25", "p50", "p75", "p90", "p95", "iqr", "spread"), qs):
        out[f"normpct_{name}"] = val
    return out


def shape_drift(doc: _Doc) -> dict[str, float]:
    """First-third vs last-third contrast: does the doc drift in DCT space?"""
    if doc.n < 3:
        return {f"drift_{k}": math.nan for k in ("cos", "adj", "ratio", "norm")}
    third = max(1, doc.n // 3)
    first_v = np.mean(doc.vectors[:third], axis=0)
    last_v = np.mean(doc.vectors[-third:], axis=0)
    denom = float(np.linalg.norm(first_v) * np.linalg.norm(last_v))
    ratios = [b[1] / b[0] if b[0] else 0.0 for b in doc.band_norms]
    norms = doc.norms()
    adj_first = doc.adjacent[: max(1, third - 1)]
    adj_last = doc.adjacent[-max(1, third - 1) :]
    return {
        "drift_cos": float(first_v @ last_v / denom) if denom else 0.0,
        "drift_adj": float(np.mean(adj_first) - np.mean(adj_last)),
        "drift_ratio": float(np.mean(ratios[:third]) - np.mean(ratios[-third:])),
        "drift_norm": float(np.median(norms[:third]) - np.median(norms[-third:])),
    }


def shape_acosq(doc: _Doc) -> dict[str, float]:
    """Adjacent-cosine quantiles: the tail of local smoothness, not its mean."""
    out: dict[str, float] = {}
    qs = _quantiles(doc.adjacent)
    for name, val in zip(("p05", "p10", "p25", "p50", "p75", "p90", "p95", "iqr", "spread"), qs):
        out[f"acosq_{name}"] = val
    return out


SHAPES = {
    "base": shape_base,
    "paircos": shape_paircos,
    "bands": shape_bands,
    "normpct": shape_normpct,
    "drift": shape_drift,
    "acosq": shape_acosq,
}


# ------------------------------------------------------------------ pipeline
def featurize_buckets(
    buckets: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[str]]]:
    """(shape -> bucket -> matrix, shape -> feature names), cached (pure fns)."""
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        if int(cached["version"]) == FEATURE_VERSION:
            ok = all(
                list(cached[f"ids_{b}"]) == list(buckets[b].id) for b in buckets
            )
            if ok:
                print(f"feature cache hit: {CACHE} (v{FEATURE_VERSION})")
                names = {s: list(cached[f"names_{s}"]) for s in SHAPES}
                feats = {
                    s: {b: cached[f"feats_{s}_{b}"] for b in buckets} for s in SHAPES
                }
                return feats, names
        print("feature cache stale -- recomputing")

    feats: dict[str, dict[str, list]] = {s: {b: [] for b in buckets} for s in SHAPES}
    names: dict[str, list[str]] = {}
    for bucket, sub in buckets.items():
        t0 = time.time()
        for i, text in enumerate(sub.generation):
            doc = _Doc(str(text))
            for shape, fn in SHAPES.items():
                row = fn(doc)
                if bucket == "A" and i == 0:
                    names.setdefault(shape, list(row))
                feats[shape][bucket].append([row[k] for k in names[shape]])
            if (i + 1) % 500 == 0:
                print(f"  {bucket}: {i + 1}/{len(sub)} ({time.time() - t0:.0f}s)")
        print(f"bucket {bucket} featurized in {time.time() - t0:.0f}s")

    out = {s: {b: np.array(rows, dtype=float) for b, rows in per_b.items()} for s, per_b in feats.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": np.array(FEATURE_VERSION)}
    for b, sub in buckets.items():
        payload[f"ids_{b}"] = np.array(list(sub.id))
    for s in SHAPES:
        payload[f"names_{s}"] = np.array(names[s])
        for b in buckets:
            payload[f"feats_{s}_{b}"] = out[s][b]
    np.savez(CACHE, **payload)
    print(f"feature cache written: {CACHE}")
    return out, names


def evaluate(ai: np.ndarray, hu: np.ndarray) -> tuple[float, dict]:
    return auroc(list(ai), list(hu)), tpr_at_fpr(list(ai), list(hu))


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    cached = np.load("data/derived/base_scores.npz")
    labels = {}
    for name, sub in buckets.items():
        labels[name] = (sub.model != "human").to_numpy(int)
        assert np.array_equal(labels[name], cached[f"labels_{name}"]), f"bucket {name} misaligned"
        print(f"bucket {name}: {len(sub)} docs ({int(labels[name].sum())} ai) -- labels aligned")

    feats, names = featurize_buckets(buckets)

    def train_scores(matrix_by_bucket: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        model = HistGradientBoostingClassifier(random_state=7)
        model.fit(matrix_by_bucket["A"], labels["A"])
        return {b: model.predict_proba(matrix_by_bucket[b])[:, 1] for b in ("B", "C")}

    # ---- shape selection on B (C stays untouched until the final pass) ----
    print("\n== solo shapes, trained on A, selected on B ==")
    candidates = list(SHAPES) + ["all", "nobase"]
    combo = {
        "all": list(SHAPES),
        "nobase": [s for s in SHAPES if s != "base"],
    }
    scores: dict[str, dict[str, np.ndarray]] = {}
    board = []
    for cand in candidates:
        if cand in combo:
            mats = {b: np.concatenate([feats[s][b] for s in combo[cand]], axis=1) for b in buckets}
        else:
            mats = feats[cand]
        scores[cand] = train_scores(mats)
        roc, res = evaluate(scores[cand]["B"][labels["B"] == 1], scores[cand]["B"][labels["B"] == 0])
        board.append((cand, roc, res["tpr"]))
        print(f"  {cand:<8} n_feat {mats['A'].shape[1]:>3} | B AUROC {roc:.3f} | B TPR@1e-3 {res['tpr']:.3f}")

    best = max(board, key=lambda row: (row[2], row[1]))
    print(f"\nselected on B: {best[0]} (TPR {best[2]:.3f}, AUROC {best[1]:.3f})")

    # ---- final pass: numbers on C, once ----
    print("\n== FINAL on C (one pass) ==")
    mine = scores[best[0]]["C"]
    roc, res = evaluate(mine[labels["C"] == 1], mine[labels["C"] == 0])
    print(f"  dct-{best[0]:<5} solo  C AUROC {roc:.3f} | C TPR@1e-3 {res['tpr']:.3f} "
          f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")

    base3 = ["relative-burst", "qgram12", "exemplar"]
    stacks = {"3det(control)": base3, "4det(ours)": base3 + ["dct"]}
    for label, members in stacks.items():
        zb = np.column_stack([cached[f"{m}_B"] for m in base3] + ([scores[best[0]]["B"]] if "dct" in members else []))
        zc = np.column_stack([cached[f"{m}_C"] for m in base3] + ([mine] if "dct" in members else []))
        meta = HistGradientBoostingClassifier(random_state=7)
        meta.fit(zb, labels["B"])
        s = meta.predict_proba(zc)[:, 1]
        roc, res = evaluate(s[labels["C"] == 1], s[labels["C"] == 0])
        print(f"  stack {label:<13} C AUROC {roc:.3f} | C TPR@1e-3 {res['tpr']:.3f} "
              f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")

    # per-shape C table for the doc (same single pass, no re-selection)
    print("\n== per-shape C table (reporting only; selection was on B) ==")
    for cand in candidates:
        roc, res = evaluate(scores[cand]["C"][labels["C"] == 1], scores[cand]["C"][labels["C"] == 0])
        print(f"  {cand:<8} C AUROC {roc:.3f} | C TPR@1e-3 {res['tpr']:.3f}")

    print(f"\nfeature names of winner: {names[best[0]] if best[0] in names else 'concat of ' + str(combo[best[0]])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
