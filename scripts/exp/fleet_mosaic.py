"""FLEET MOSAIC — the qg_s256_ck2 series, RAW vs digested (bitter-lesson test).

The wired feature is mean(series); the series is 256 exchangeable long-range
self-similarity scores per doc. Donk: emit the raw vector (Sutton), with the
sorted-series percentile profile as the digested control.

Arms (train A, read C): panel253, panel+raw256, raw256 alone,
panel+sorted19, sorted19 alone.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.burst import random_change_series
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_mosaic.md"
SAMPLES, WINDOW, MIN_GAP = 256, 150, 50
PCTS = np.linspace(5, 95, 19)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    raw, srt = {}, {}
    for b in "ABC":
        r_rows, s_rows = [], []
        for t in buckets[b].generation:
            s = random_change_series(str(t), window=WINDOW, samples=SAMPLES,
                                     min_gap=MIN_GAP, metric="ck2", unit="tokens")
            if len(s) < SAMPLES:
                s = s + [np.nan] * (SAMPLES - len(s))
            r_rows.append(s)
            srt_arr = np.sort(np.array(s, dtype=float))
            s_rows.append(np.nanpercentile(srt_arr, PCTS))
        raw[b] = np.array(r_rows, dtype=float)
        srt[b] = np.array(s_rows, dtype=float)
        print(f"{b} done", flush=True)

    panel = np.load("data/derived/full_features.npz")
    n_panel = len(panel["feature_names"])
    Xp = {b: panel[f"X_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(Xp["A"], axis=0))

    def prep(X, m):
        X = X.copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(m, bad[1])
        return X

    p_means = means
    r_means = np.nan_to_num(np.nanmean(raw["A"], axis=0))
    s_means = np.nan_to_num(np.nanmean(srt["A"], axis=0))

    arms = {
        f"panel{n_panel}": lambda b: prep(Xp[b], p_means),
        f"panel+raw{SAMPLES}": lambda b: np.column_stack(
            [prep(Xp[b], p_means), prep(raw[b], r_means)]),
        f"raw{SAMPLES} alone": lambda b: prep(raw[b], r_means),
        "panel+sorted19": lambda b: np.column_stack(
            [prep(Xp[b], p_means), prep(srt[b], s_means)]),
        "sorted19 alone": lambda b: prep(srt[b], s_means),
    }

    lines = ["# FLEET MOSAIC — raw s256 series vs digested (train A, read C)\n\n",
             "| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |\n|---|---|---|---|---|\n"]
    for arm, get in arms.items():
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(get("A"), labels["A"])
        s = m.predict_proba(get("C"))[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        r1 = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]), fpr=1e-2)
        r3 = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]), fpr=1e-3)
        lines.append(f"| {arm} | {get('A').shape[1]} | {roc:.4f} | "
                     f"{r1['tpr']:.3f} [{r1['tpr_lo']:.3f},{r1['tpr_hi']:.3f}] | "
                     f"{r3['tpr']:.3f} |\n")
        print(lines[-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
