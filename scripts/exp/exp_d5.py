"""D5: does calibrating the DCT score fix the 4-detector stack's low-FPR corner?

Hypothesis: DCT AUROC is fine-ish (0.654) but its score TAIL is miscalibrated,
so stacking it raw dilutes TPR@FPR=1e-3 (3-det 0.170 -> 4-det 0.155).

Protocol (RULES #4, same as the fleet): source-disjoint A/B/C buckets via
evaldata.split_buckets. DCT detector = HistGradientBoostingClassifier(7)
trained on A. Calibrators fit on B ONLY; meta-learner trains on B; all final
numbers on C, the untouched eval. Base-detector scores come from
data/derived/base_scores.npz (cached A scores ignored — in-sample).

Calibrations compared:
  raw           — identity
  platt         — logistic regression on the B scores (a)
  isotonic      — sklearn IsotonicRegression on B (b)
  quantile      — empirical-CDF map to (0,1), per detector, fitted on B (c)
  logit         — parameter-free logit(clip(p, 1e-4, 1-1e-4)); expands the
                  compressed high tail where the FPR=1e-3 threshold lives (d)

Run (from the main-repo cwd, data lives there):
  set PYTHONPATH=<worktree>\\src && .venv\\Scripts\\python <worktree>\\scripts\\exp_d5.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_text_detection import dct
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

BASE_DETS = ("relative-burst", "qgram12", "exemplar")
ALL_DETS = BASE_DETS + ("dct",)
EPS = 1e-4


# --- calibrator factories: fit on B scores -> transform fn applied to B/C ---
def identity_fit(scores, labels):
    return lambda x: x


def platt_fit(scores, labels):
    lr = LogisticRegression(max_iter=2000).fit(scores.reshape(-1, 1), labels)
    return lambda x: lr.predict_proba(x.reshape(-1, 1))[:, 1]


def isotonic_fit(scores, labels):
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    return lambda x: iso.predict(x)


def quantile_fit(scores, labels):
    xs = np.sort(scores)
    qs = (np.arange(len(xs)) + 0.5) / len(xs)
    return lambda x: np.interp(x, xs, qs)


def logit_fit(scores, labels):
    def transform(x):
        p = np.clip(x, EPS, 1 - EPS)
        return np.log(p / (1 - p))

    return transform


def dct_matrix(texts, col_means=None):
    rows = []
    for text in texts:
        feats = dct.dct_features(str(text))
        rows.append([feats[name] for name in dct.DCT_FEATURE_NAMES])
    X = np.array(rows, dtype=float)
    if col_means is None:
        col_means = np.nanmean(X, axis=0)
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(col_means, bad[1])
    return X, col_means


def report(tag, labels, scores):
    roc = auroc(list(scores[labels == 1]), list(scores[labels == 0]))
    res = tpr_at_fpr(list(scores[labels == 1]), list(scores[labels == 0]))
    print(
        f"  {tag:<26} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} "
        f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]",
        flush=True,
    )
    return {"auroc": roc, "tpr": res["tpr"], "lo": res["tpr_lo"], "hi": res["tpr_hi"]}


def main():
    t0 = time.time()
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    store = np.load("data/derived/base_scores.npz")
    labels = {}
    for name, sub in buckets.items():
        labels[name] = (sub.model != "human").to_numpy(int)
        if name in ("B", "C"):
            assert np.array_equal(labels[name], store[f"labels_{name}"]), (
                f"bucket {name} labels misaligned with cached scores"
            )
        print(f"bucket {name}: {len(sub)} docs ({int((sub.model == 'human').sum())} human)", flush=True)

    # --- DCT detector: featurize A/B/C, train on A, score B/C ---
    feats, col_means = {}, None
    for name in ("A", "B", "C"):
        t = time.time()
        feats[name], col_means = dct_matrix(buckets[name].generation, col_means)
        print(f"  dct featurized {name} in {time.time() - t:.1f}s", flush=True)
    model = HistGradientBoostingClassifier(random_state=7)
    model.fit(feats["A"], labels["A"])

    raw_b = {det: store[f"{det}_B"] for det in BASE_DETS}
    raw_c = {det: store[f"{det}_C"] for det in BASE_DETS}
    raw_b["dct"] = model.predict_proba(feats["B"])[:, 1]
    raw_c["dct"] = model.predict_proba(feats["C"])[:, 1]

    print("\n== single detectors on C (raw) ==", flush=True)
    for det in ALL_DETS:
        report(f"{det}", labels["C"], raw_c[det])

    # --- the miscalibration hypothesis itself: who owns DCT's top tail on B?
    hu, ai = raw_b["dct"][labels["B"] == 0], raw_b["dct"][labels["B"] == 1]
    top = np.quantile(hu, 0.999)
    print("\n== dct tail on B (raw hgb prob) ==", flush=True)
    print(f"  human p50 {np.median(hu):.3f} | p99 {np.quantile(hu, 0.99):.3f} | p99.9 {top:.3f}", flush=True)
    print(f"  ai    p50 {np.median(ai):.3f} | share of AI above human p99.9: {(ai >= top).mean():.3f}", flush=True)

    conditions = {
        "3det raw": (BASE_DETS, {}),
        "4det raw": (ALL_DETS, {}),
        "4det platt(dct) (a)": (ALL_DETS, {"dct": platt_fit}),
        "4det isotonic(dct) (b)": (ALL_DETS, {"dct": isotonic_fit}),
        "4det quantile(all) (c)": (ALL_DETS, {d: quantile_fit for d in ALL_DETS}),
        "4det logit(dct) (d)": (ALL_DETS, {"dct": logit_fit}),
        "4det platt(all) (bonus)": (ALL_DETS, {d: platt_fit for d in ALL_DETS}),
    }
    metas = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(random_state=7),
    }

    results = {}
    print("\n== stacks: calibrators fit on B, meta trained on B, evaluated on C ==", flush=True)
    for cond, (dets, calibrators) in conditions.items():
        zb_cols, zc_cols = [], []
        for det in dets:
            transform = calibrators.get(det, identity_fit)(raw_b[det], labels["B"])
            zb_cols.append(transform(raw_b[det]))
            zc_cols.append(transform(raw_c[det]))
        zb, zc = np.column_stack(zb_cols), np.column_stack(zc_cols)
        for meta_name, meta in metas.items():
            meta.fit(zb, labels["B"])
            scores = meta.predict_proba(zc)[:, 1]
            results[f"{cond} / {meta_name}"] = report(f"{cond} / {meta_name}", labels["C"], scores)

    print(f"\ntotal runtime {time.time() - t0:.1f}s", flush=True)
    best = max(results, key=lambda k: results[k]["tpr"])
    r = results[best]
    print(f"best tail: {best} TPR {r['tpr']:.3f} AUROC {r['auroc']:.3f}", flush=True)
    print(f"clears 0.170 TPR: {r['tpr'] > 0.170} | holds 0.896 AUROC: {r['auroc'] >= 0.896 - 0.005}", flush=True)


if __name__ == "__main__":
    main()
