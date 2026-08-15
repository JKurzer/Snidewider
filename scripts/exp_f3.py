"""F3 experiment: best q-gram parameterization / feature expansion.

Protocol (fleet-wide): RAID dev fold only, all 2000 humans + 4000 AI
(random_state=17), 50/50 source-disjoint split (RandomState(23)).

Stages:
  1. Featurize docs into a candidate matrix (cached to data/derived —
     features are pure functions of text, cache is just an optimization).
  2. Per-feature AUROC on the test half: q sweep {2,3,4,5} for the profile
     stats, collision-spectrum percentiles, and ck2/qgram/bag change series.
  3. Combined logreg/HGB for the baseline 9 vs the recommended qgram set,
     on the full (length-biased) subset AND a length-matched subset.

Run:  set PYTHONPATH=...\\src&& .venv\\Scripts\\python scripts\\exp_f3.py
"""

from __future__ import annotations

import math
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ai_text_detection
from ai_text_detection import burst, qgram
from ai_text_detection.metrics import auroc, tpr_at_fpr

DATA = Path(r"C:\Users\poly\ai-text-detection\data\derived\raid_splits.parquet")
CACHE = Path(__file__).resolve().parent.parent / "data" / "derived" / "f3_features.parquet"
N_AI = 4000
QS = (2, 3, 4, 5)

# Baseline 9 (features.py FEATURE_NAMES), renamed to this script's columns.
BASELINE9 = [
    "short_ck2_mean",
    "short_ck2_stdev",
    "mid_ck2_mean",
    "mid_ck2_stdev",
    "q3_total",
    "q3_distinct_ratio",
    "q3_repeat_frac",
    "q3_max_share",
    "q3_entropy",
]

# F3 recommendation — filled in from the stage-2 table (see docs/exp_f3.md).
# Deliberately excludes q3_total: it is a document-length proxy and collapses
# under length matching (length leak, not a real signal).
# F3 recommendation (honest, tie-corrected AUROCs — see docs/exp_f3.md):
# mid_qgram_mean is the strongest single feature (sep 0.38); profile stats
# are weak individually but add up. q3_total dropped: length leak.
RECOMMENDED = [
    "short_ck2_mean",
    "short_ck2_stdev",
    "mid_ck2_mean",
    "mid_ck2_stdev",
    "mid_qgram_mean",
    "mid_qgram_stdev",
    "q2_entropy",
    "q3_entropy",
    "q3_distinct_ratio",
    "q3_repeat_frac",
    "q3_max_share",
    "q5_top10_share",
]

# Ablation: does the qgram metric add anything on top of the baseline 9?
BASE_PLUS_QSERIES = BASELINE9 + ["mid_qgram_mean", "mid_qgram_stdev"]

# Ablation: change series alone, both metrics, no profile stats.
SERIES_ONLY = [
    "short_ck2_mean",
    "short_ck2_stdev",
    "mid_ck2_mean",
    "mid_ck2_stdev",
    "mid_qgram_mean",
    "mid_qgram_stdev",
]

# Profile-expanded: no NaN-prone series -> covers ALL docs, including the
# short ones the baseline has to drop (81% of docs!). The coverage play.
PROFILE_EXP = [
    "q2_entropy",
    "q3_entropy",
    "q4_entropy",
    "q3_distinct_ratio",
    "q3_repeat_frac",
    "q3_max_share",
    "q4_max_share",
    "q5_top10_share",
    "q4_count_p50",
    "q5_count_p50",
]

MODEL_SETS = {
    "baseline9": BASELINE9,
    "baseline+qgram-series": BASE_PLUS_QSERIES,
    "series-only": SERIES_ONLY,
    "recommended": RECOMMENDED,
    "profile-exp": PROFILE_EXP,
}

PROFILE_STATS = ("total", "distinct_ratio", "repeat_frac", "max_share", "entropy")
SPECTRUM_STATS = ("count_p50", "count_p90", "count_p99", "top10_share")


def profile_features(text_bytes: bytes, q: int) -> dict[str, float]:
    """All profile + collision-spectrum stats for one q. NaNs when len < q."""
    names = [f"q{q}_{s}" for s in PROFILE_STATS + SPECTRUM_STATS]
    profile = qgram.profile(text_bytes, q)
    total = sum(c for _, c in profile)
    if total == 0:
        return {name: math.nan for name in names}
    counts = np.array([c for _, c in profile], dtype=np.float64)
    distinct = len(counts)
    probs = counts / total
    desc = np.sort(counts)[::-1]
    return {
        f"q{q}_total": float(total),
        f"q{q}_distinct_ratio": distinct / total,
        f"q{q}_repeat_frac": float(np.mean(counts > 1)),
        f"q{q}_max_share": float(counts.max() / total),
        f"q{q}_entropy": float(-np.sum(probs * np.log2(probs))),
        f"q{q}_count_p50": float(np.percentile(counts, 50)),
        f"q{q}_count_p90": float(np.percentile(counts, 90)),
        f"q{q}_count_p99": float(np.percentile(counts, 99)),
        f"q{q}_top10_share": float(desc[:10].sum() / total),
    }


def series_features(text: str) -> dict[str, float]:
    """Change-series summaries: short-range bytes + midrange tokens, 3 metrics."""
    feats: dict[str, float] = {}
    for metric in ("ck2", "qgram", "bag"):
        short = burst.burst_features(text, window=64, gap=0, unit="bytes", metric=metric)
        feats[f"short_{metric}_mean"] = short["mean"]
        feats[f"short_{metric}_stdev"] = short["stdev"]
        mid = burst.burst_features(
            text, window=150, samples=32, min_gap=50,
            unit="tokens", mode="random", metric=metric,
        )
        feats[f"mid_{metric}_mean"] = mid["mean"]
        feats[f"mid_{metric}_stdev"] = mid["stdev"]
    return feats


def extract(text: str) -> dict[str, float]:
    text_bytes = text.encode("utf-8")
    feats: dict[str, float] = {"length": float(len(text_bytes))}
    for q in QS:
        feats.update(profile_features(text_bytes, q))
    feats.update(series_features(text))
    return feats


def build_features() -> pd.DataFrame:
    """Dev-fold feature matrix; cached (pure functions, RULES #5)."""
    if CACHE.exists():
        print(f"loading cached features: {CACHE}")
        return pd.read_parquet(CACHE)
    df = pd.read_parquet(DATA)
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"]
    ai = dev[dev.model != "human"].sample(n=N_AI, random_state=17)
    docs = [(str(t), 0, s) for t, s in zip(humans.generation, humans.source_id)]
    docs += [(str(t), 1, s) for t, s in zip(ai.generation, ai.source_id)]
    print(f"featurizing {len(docs)} docs ({len(humans)} human / {len(ai)} ai)...")
    rows, t0 = [], time.time()
    for i, (text, label, source) in enumerate(docs):
        rows.append({"label": label, "source_id": source, **extract(text)})
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(docs)} ({rate:.0f} docs/s, "
                  f"eta {(len(docs) - i - 1) / rate:.0f}s)")
    out = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    print(f"cached -> {CACHE} ({time.time() - t0:.0f}s)")
    return out


def split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """50/50 source-disjoint split, RandomState(23) — fleet protocol."""
    sources = df.source_id.to_numpy()
    unique = np.unique(sources)
    train_sources = set(
        np.random.RandomState(23).choice(unique, len(unique) // 2, replace=False)
    )
    train = np.array([s in train_sources for s in sources])
    return train, ~train


def per_feature_auroc(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """AUROC per column on the test half (NaN rows dropped per column)."""
    _, test = split_masks(df)
    te = df[test]
    ai, hu = te.label == 1, te.label == 0
    out = {}
    for col in cols:
        ok = te[col].notna()
        if ok.sum() == 0:
            out[col] = math.nan
            continue
        out[col] = auroc(list(te.loc[ok & ai, col]), list(te.loc[ok & hu, col]))
    return pd.Series(out)


def length_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Greedy nearest-length AI match per human (no replacement)."""
    humans = df[df.label == 0]
    pool = sorted((r.length, i) for i, r in df[df.label == 1].iterrows())
    picked = []
    for _, h in humans.sample(frac=1.0, random_state=5).iterrows():
        pos = bisect_left(pool, (h.length, -1))
        best = min(
            (p for p in (pos - 1, pos) if 0 <= p < len(pool)),
            key=lambda p: abs(pool[p][0] - h.length),
        )
        picked.append(pool.pop(best)[1])
    return pd.concat([humans, df.loc[picked]])


def eval_models(df: pd.DataFrame, cols: list[str], tag: str) -> None:
    """Logreg + HGB on source-disjoint split; AUROC + TPR@1e-3 (Wilson)."""
    usable = df.dropna(subset=cols)
    train, test = split_masks(usable)
    Xtr, ytr = usable.loc[train, cols].to_numpy(), usable.loc[train, "label"].to_numpy()
    Xte, yte = usable.loc[test, cols].to_numpy(), usable.loc[test, "label"].to_numpy()
    print(f"[{tag}] usable {len(usable)}/{len(df)} "
          f"(train {len(Xtr)} / test {len(Xte)}, source-disjoint)")
    for name, model in (
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("hgb", HistGradientBoostingClassifier(random_state=23)),
    ):
        model.fit(Xtr, ytr)
        scores = model.predict_proba(Xte)[:, 1]
        res = tpr_at_fpr(list(scores[yte == 1]), list(scores[yte == 0]))
        print(f"  {name:<7} AUROC {auroc(list(scores[yte == 1]), list(scores[yte == 0])):.3f}"
              f" | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")


def class_medians(df: pd.DataFrame, col: str) -> str:
    ai = df.loc[df.label == 1, col].median()
    hu = df.loc[df.label == 0, col].median()
    return f"{col}: median ai={ai:.2f} human={hu:.2f}"


def main() -> None:
    print(f"ai_text_detection from: {ai_text_detection.__file__}")
    df = build_features()
    feature_cols = [c for c in df.columns if c not in ("label", "source_id", "length")]
    print(f"matrix: {df.shape}, labels: {df.label.value_counts().to_dict()}")

    aucs = per_feature_auroc(df, feature_cols)

    print("\n== q sweep: profile stats (test-half AUROC) ==")
    sweep = pd.DataFrame(
        {stat: {q: aucs.get(f"q{q}_{stat}") for q in QS} for stat in PROFILE_STATS}
    ).T
    sweep.columns = [f"q={q}" for q in QS]
    print(sweep.round(3).to_string())

    print("\n== q sweep: collision spectrum (test-half AUROC) ==")
    spec = pd.DataFrame(
        {stat: {q: aucs.get(f"q{q}_{stat}") for q in QS} for stat in SPECTRUM_STATS}
    ).T
    spec.columns = [f"q={q}" for q in QS]
    print(spec.round(3).to_string())

    print("\n== change series: metric variants (test-half AUROC) ==")
    series_cols = [c for c in feature_cols if c.startswith(("short_", "mid_"))]
    for name in sorted(series_cols):
        print(f"  {name:<20} {aucs[name]:.3f}")

    print("\n== top 12 candidates by separation |AUROC - 0.5| ==")
    by_sep = aucs.reindex((aucs - 0.5).abs().sort_values(ascending=False).index)
    for name, val in by_sep.head(12).items():
        print(f"  {name:<20} {val:.3f}  (sep {abs(val - 0.5):.3f})")

    _, ftest = split_masks(df)
    fte = df[ftest]
    print("\n== tie-rate check: P(count_p50 > 1) by class (full test half) ==")
    for q in QS:
        col = f"q{q}_count_p50"
        ai_rate = (fte.loc[fte.label == 1, col] > 1).mean()
        hu_rate = (fte.loc[fte.label == 0, col] > 1).mean()
        print(f"  {col:<14} ai {ai_rate:.3f} | human {hu_rate:.3f}")

    matched = length_matched(df)
    print(f"\nlength-matched subset: {len(matched)} docs "
          f"(mean len ai {matched[matched.label == 1].length.mean():.0f} vs "
          f"human {matched[matched.label == 0].length.mean():.0f})")
    _, mtest = split_masks(matched)
    mte = matched[mtest]
    len_auc = auroc(list(mte.loc[mte.label == 1, "length"]),
                    list(mte.loc[mte.label == 0, "length"]))
    print(f"length-as-feature AUROC on matched test half: {len_auc:.3f} "
          "(~0.5 = matching worked)")

    print("\n== diagnostics: full subset ==")
    for col in ("q5_count_p50", "q4_count_p50", "q3_count_p50", "q5_count_p90"):
        print(f"  {class_medians(df, col)}")

    m_aucs = per_feature_auroc(matched, feature_cols)
    print("\n== q sweep on LENGTH-MATCHED subset: spectrum + profile ==")
    msweep = pd.DataFrame(
        {stat: {q: m_aucs.get(f"q{q}_{stat}") for q in QS}
         for stat in SPECTRUM_STATS + PROFILE_STATS}
    ).T
    msweep.columns = [f"q={q}" for q in QS]
    print(msweep.round(3).to_string())
    print("\n== series on length-matched subset ==")
    for name in sorted(series_cols):
        print(f"  {name:<20} {m_aucs[name]:.3f}")

    print("\n== models: full subset (length bias intact) ==")
    for tag, cols in MODEL_SETS.items():
        eval_models(df, cols, f"{tag}/full")
    print("\n== models: length-matched subset ==")
    for tag, cols in MODEL_SETS.items():
        eval_models(matched, cols, f"{tag}/matched")


if __name__ == "__main__":
    main()
