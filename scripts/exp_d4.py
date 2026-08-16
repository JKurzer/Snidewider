"""D4: pooling-strategy sweep for DCT doc features (tail-signal hypothesis).

dct.dct_features pools per-segment quantities (adjacent-sentence cosines,
order-energy ratios ||c[1]||/||c[0]||) with mean/std over ALL segments.
Hypothesis: mean-pooling flattens the tail — the extremes of the per-segment
distribution may separate AI from human better than the average.

We replicate dct_features' per-segment loop once per doc (embed_sentence +
dct_coefficients, k=2), keep the raw per-segment arrays, and sweep pooling
strategies: meanstd (reference), trimmed, quantiles, min/max, top-5 energetic
segments, decile tails, and combos.

Protocol (leakage-proof, RULES #4): detector = HistGradientBoostingClassifier
(random_state=7) trained on bucket A. Pooling selected on bucket B via meta
out-of-fold stack scores. Final numbers ONCE on bucket C. The other three
detectors' scores come from data/derived/base_scores.npz (row-aligned with
evaldata.split_buckets; alignment verified by label equality).

Run (from the main checkout so data/ paths resolve):
  set PYTHONPATH=<worktree>\\src && .venv\\Scripts\\python <worktree>\\scripts\\exp_d4.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import ai_text_detection
from ai_text_detection.dct import dct_coefficients, embed_sentence
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

assert "d4-pooling" in ai_text_detection.__file__, ai_text_detection.__file__
print(f"import check: {ai_text_detection.__file__}")

WORKTREE = Path(__file__).resolve().parents[1]
CACHE = WORKTREE / "data" / "derived" / "d4_segments.npz"
BASE_SCORES = Path("data/derived/base_scores.npz")  # cwd = main checkout
PARQUET = Path("data/derived/raid_splits.parquet")
BASE_DETECTORS = ["relative-burst", "qgram12", "exemplar"]
RS = 7


# ---------------------------------------------------------------- per-doc raw
def per_segment_quantities(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent-cosine + order-energy arrays; dct_features' loop, unpooled."""
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 2]
    vectors = []
    energies = []
    for sentence in sentences:
        embedded = embed_sentence(sentence)
        if embedded.shape[0] < 2:
            continue
        coeffs = dct_coefficients(embedded)  # k=2, paper setting
        vectors.append(coeffs.reshape(-1).astype(np.float64))
        base = float(np.linalg.norm(coeffs[0]))
        energies.append(float(np.linalg.norm(coeffs[1]) / base) if base else 0.0)
    cosines = []
    for i in range(len(vectors) - 1):
        denom = float(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i + 1]))
        cosines.append(float(vectors[i] @ vectors[i + 1] / denom) if denom else 0.0)
    return np.asarray(cosines), np.asarray(energies)


def featurize_buckets(buckets: dict[str, pd.DataFrame]) -> dict[str, list]:
    """Raw per-segment arrays per doc; cached (pure functions, RULES #5)."""
    if CACHE.exists():
        print(f"loading cached segments from {CACHE}")
        z = np.load(CACHE)
        out = {}
        for name in buckets:
            cos = [z[f"{name}_cos"][z[f"{name}_cptr"][i] : z[f"{name}_cptr"][i + 1]]
                   for i in range(len(buckets[name]))]
            en = [z[f"{name}_en"][z[f"{name}_eptr"][i] : z[f"{name}_eptr"][i + 1]]
                  for i in range(len(buckets[name]))]
            out[name] = list(zip(cos, en))
        return out
    out = {}
    for name, sub in buckets.items():
        t0 = time.time()
        rows = [per_segment_quantities(str(t)) for t in sub.generation]
        out[name] = rows
        print(f"  featurized bucket {name}: {len(rows)} docs in {time.time() - t0:.0f}s")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for name, rows in out.items():
        cos_l = [r[0] for r in rows]
        en_l = [r[1] for r in rows]
        flat[f"{name}_cos"] = np.concatenate(cos_l) if cos_l else np.array([])
        flat[f"{name}_cptr"] = np.concatenate([[0], np.cumsum([len(c) for c in cos_l])])
        flat[f"{name}_en"] = np.concatenate(en_l) if en_l else np.array([])
        flat[f"{name}_eptr"] = np.concatenate([[0], np.cumsum([len(e) for e in en_l])])
    np.savez(CACHE, **flat)
    print(f"cached segments -> {CACHE}")
    return out


# ------------------------------------------------------------------- poolings
def _tmean(x: np.ndarray, frac: float = 0.1) -> float:
    """Trimmed mean: drop the frac tails (10/90 default)."""
    if len(x) == 0:
        return np.nan
    k = int(len(x) * frac)
    core = np.sort(x)[k : len(x) - k]
    return float(core.mean() if len(core) else x.mean())


def _qs(x: np.ndarray) -> list[float]:
    if len(x) == 0:
        return [np.nan] * 3
    return [float(v) for v in np.quantile(x, [0.1, 0.5, 0.9])]


def _tails(x: np.ndarray, frac: float = 0.1) -> list[float]:
    """Mean of the bottom / top decile of the per-segment distribution."""
    if len(x) == 0:
        return [np.nan, np.nan]
    k = max(1, int(len(x) * frac))
    s = np.sort(x)
    return [float(s[:k].mean()), float(s[-k:].mean())]


def _agg(x: np.ndarray, fn) -> float:
    return float(fn(x)) if len(x) else np.nan


def _top5(cos: np.ndarray, en: np.ndarray) -> list[float]:
    """Pool over the top-5 most order-energetic segments only."""
    if len(en) == 0:
        return [np.nan] * 4
    idx = set(int(i) for i in np.argsort(en)[-5:])
    boundary = np.asarray([c for i, c in enumerate(cos) if i in idx or (i + 1) in idx])
    if len(boundary) == 0:
        boundary = cos
    return [
        float(en[sorted(idx)].mean()),
        float(en[sorted(idx)].std()),
        _agg(boundary, np.mean),
        _agg(boundary, np.std),
    ]


def _extremes(cos: np.ndarray, en: np.ndarray) -> list[float]:
    """Extreme-quantile combo: q10/q90 and the decile range of each quantity."""
    qc, qe = _qs(cos), _qs(en)
    return [qc[0], qc[2], qc[2] - qc[0], qe[0], qe[2], qe[2] - qe[0]]


def _named(names: list[str]):
    def wrap(fn):
        fn.feature_names = names
        return fn
    return wrap


POOLINGS = {
    "meanstd": _named(["cos_mean", "cos_std", "en_mean", "en_std"])(
        lambda c, e: [_agg(c, np.mean), _agg(c, np.std),
                      _agg(e, np.mean), _agg(e, np.std)]),
    "trimmed": _named(["cos_tmean", "cos_std", "en_tmean", "en_std"])(
        lambda c, e: [_tmean(c), _agg(c, np.std), _tmean(e), _agg(e, np.std)]),
    "quantiles": _named(["cos_q10", "cos_q50", "cos_q90",
                         "en_q10", "en_q50", "en_q90"])(
        lambda c, e: _qs(c) + _qs(e)),
    "max": _named(["cos_max", "en_max"])(lambda c, e: [_agg(c, np.max), _agg(e, np.max)]),
    "min": _named(["cos_min", "en_min"])(lambda c, e: [_agg(c, np.min), _agg(e, np.min)]),
    "minmax": _named(["cos_min", "cos_max", "en_min", "en_max"])(
        lambda c, e: [_agg(c, np.min), _agg(c, np.max),
                      _agg(e, np.min), _agg(e, np.max)]),
    "top5_energy": _named(["top5_en_mean", "top5_en_std",
                           "top5_cos_mean", "top5_cos_std"])(_top5),
    "tails10": _named(["cos_lo10", "cos_hi10", "en_lo10", "en_hi10"])(
        lambda c, e: _tails(c) + _tails(e)),
    "extremes": _named(["cos_q10", "cos_q90", "cos_range",
                        "en_q10", "en_q90", "en_range"])(_extremes),
    "meanstd+tails": _named(["cos_mean", "cos_std", "en_mean", "en_std",
                             "cos_lo10", "cos_hi10", "en_lo10", "en_hi10"])(
        lambda c, e: [_agg(c, np.mean), _agg(c, np.std),
                      _agg(e, np.mean), _agg(e, np.std)] + _tails(c) + _tails(e)),
    "all": _named(["cos_mean", "cos_std", "en_mean", "en_std",
                   "cos_q10", "cos_q50", "cos_q90", "en_q10", "en_q50", "en_q90",
                   "cos_min", "cos_max", "en_min", "en_max",
                   "cos_lo10", "cos_hi10", "en_lo10", "en_hi10",
                   "top5_en_mean", "top5_en_std", "top5_cos_mean", "top5_cos_std"])(
        lambda c, e: [_agg(c, np.mean), _agg(c, np.std),
                      _agg(e, np.mean), _agg(e, np.std)]
        + _qs(c) + _qs(e)
        + [_agg(c, np.min), _agg(c, np.max), _agg(e, np.min), _agg(e, np.max)]
        + _tails(c) + _tails(e) + _top5(c, e)),
}


def build_matrix(rows: list, pooling, col_means: np.ndarray | None = None):
    X = np.array([pooling(c, e) for c, e in rows], dtype=float)
    if col_means is None:
        col_means = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)  # fit on A only
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_means, inds[1])
    return X, col_means


def metrics_on(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    ai, hu = list(scores[labels == 1]), list(scores[labels == 0])
    res = tpr_at_fpr(ai, hu)
    return {"auroc": auroc(ai, hu), "tpr": res["tpr"],
            "tpr_lo": res["tpr_lo"], "tpr_hi": res["tpr_hi"]}


def main() -> None:
    df = pd.read_parquet(PARQUET)
    buckets = split_buckets(df)
    base = np.load(BASE_SCORES)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}
    for b in labels:
        assert len(labels[b]) == len(base[f"labels_{b}"]), b
        assert (labels[b] == base[f"labels_{b}"]).all(), f"row misalignment in {b}"
        print(f"bucket {b}: {len(labels[b])} docs ({int((labels[b] == 0).sum())} human)")

    raw = featurize_buckets(buckets)

    # --- sweep: detector on A, select pooling on B (meta out-of-fold) ---
    print("\n== pooling sweep (detector trained on A, selection on B) ==")
    table = {}
    fitted = {}
    for name, pooling in POOLINGS.items():
        Xa, means = build_matrix(raw["A"], pooling)
        Xb, _ = build_matrix(raw["B"], pooling, means)
        Xc, _ = build_matrix(raw["C"], pooling, means)
        det = HistGradientBoostingClassifier(random_state=RS)
        det.fit(Xa, labels["A"])
        s_b, s_c = det.predict_proba(Xb)[:, 1], det.predict_proba(Xc)[:, 1]
        fitted[name] = {"B": s_b, "C": s_c}

        dct_b = metrics_on(s_b, labels["B"])
        zb = {d: base[f"{d}_B"] for d in BASE_DETECTORS} | {"dct": s_b}
        cols = [zb[d] for d in BASE_DETECTORS] + [zb["dct"]]
        oof = cross_val_predict(
            HistGradientBoostingClassifier(random_state=RS),
            np.column_stack(cols), labels["B"], method="predict_proba",
            cv=StratifiedKFold(5, shuffle=True, random_state=RS))[:, 1]
        stack_b = metrics_on(oof, labels["B"])
        table[name] = {"dct_auroc_b": dct_b["auroc"], "dct_tpr_b": dct_b["tpr"],
                       "stack_auroc_b": stack_b["auroc"], "stack_tpr_b": stack_b["tpr"]}
        print(f"  {name:<14} dct(B) AUROC {dct_b['auroc']:.3f} TPR {dct_b['tpr']:.3f} | "
              f"stack-oof(B) AUROC {stack_b['auroc']:.3f} TPR {stack_b['tpr']:.3f}")

    winner = max(table, key=lambda n: (table[n]["stack_tpr_b"], table[n]["stack_auroc_b"]))
    print(f"\nselected on B: {winner}")

    # --- final numbers ONCE on C: 3-det repro, reference, winner ---
    print("\n== final on C (meta trained on B's 4 scores) ==")
    results = {"selection_table": table, "winner": winner, "C": {}}

    def final_C(tag: str, dct_scores: dict[str, np.ndarray] | None = None):
        cols = {d: base[f"{d}_B"] for d in BASE_DETECTORS}
        cols_c = {d: base[f"{d}_C"] for d in BASE_DETECTORS}
        if dct_scores is not None:
            cols["dct"], cols_c["dct"] = dct_scores["B"], dct_scores["C"]
        meta = HistGradientBoostingClassifier(random_state=RS)
        order = BASE_DETECTORS + (["dct"] if dct_scores is not None else [])
        meta.fit(np.column_stack([cols[d] for d in order]), labels["B"])
        s = meta.predict_proba(np.column_stack([cols_c[d] for d in order]))[:, 1]
        m = metrics_on(s, labels["C"])
        results["C"][tag] = m
        print(f"  {tag:<22} AUROC {m['auroc']:.3f} | TPR@1e-3 {m['tpr']:.3f} "
              f"[{m['tpr_lo']:.3f}, {m['tpr_hi']:.3f}]")

    final_C("3det-repro (expect~.170)")
    final_C("meanstd-reference", fitted["meanstd"])
    if winner != "meanstd":
        final_C(f"WINNER:{winner}", fitted[winner])

    out = WORKTREE / "data" / "derived" / "d4_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
