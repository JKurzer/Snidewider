"""Holdout breakout: per-feature devC-vs-holdout delta table + frozen ensemble.

Second holdout contact, ordered by Donk. Same threshold protocol as the exam:
bars from C humans, applied to holdout; achieved FPR reported honestly.
Holdout features cached to data/derived/holdout_features.npz so this never
costs another featurization pass.

Usage: .venv\\Scripts\\python scripts/holdout_breakout.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ai_text_detection import qgram
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import EXEMPLAR_FEATURE_NAMES, ExemplarBank, exemplar_vector
from ai_text_detection.feature_sets import QGRAM12_NAMES, qgram12_vector, relative_vector
from ai_text_detection.features_relative import FEATURE_NAMES_RELATIVE
from ai_text_detection.metrics import auroc

N_BANK = 150
HOLDOUT_AI_SAMPLE = 20_000
OUT_NPZ = "data/derived/holdout_features.npz"
OUT_MD = "docs/holdout_breakout.md"


def main() -> None:
    dev = np.load("data/derived/full_features.npz")
    names = list(dev["feature_names"])
    Xa, ya = dev["X_A"], dev["y_A"]
    Xc, yc = dev["X_C"], dev["y_C"]

    # ---- holdout featurization (cached this time) ----
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    bucket_a = split_buckets(df)["A"]
    bank_ai = ExemplarBank.from_texts(
        [str(t) for t in bucket_a[bucket_a.model != "human"].generation[:N_BANK]]
    )
    bank_hu = ExemplarBank.from_texts(
        [str(t) for t in bucket_a[bucket_a.model == "human"].generation[:N_BANK]]
    )

    hold = df[df.fold == "holdout"]
    hold_hu = hold[hold.model == "human"]
    hold_ai = hold[hold.model != "human"].sample(n=HOLDOUT_AI_SAMPLE, random_state=97)

    if Path(OUT_NPZ).exists():
        cached = np.load(OUT_NPZ)
        store = {"human": cached["X_hu"], "ai": cached["X_ai"]}
        print(f"loaded {OUT_NPZ} (cached)")
    else:
        store = {}
        for name, sub in (("human", hold_hu), ("ai", hold_ai)):
            rows = []
            for text in sub.generation:
                text = str(text)
                tail = dct_tail_features(text)
                rows.append(
                    relative_vector(text)
                    + qgram12_vector(text)
                    + exemplar_vector(qgram.profile(text.encode("utf-8"), 3), bank_ai, bank_hu)
                    + [tail[k] for k in sorted(tail)]
                )
                if len(rows) % 2000 == 0:
                    print(f"  {name}: {len(rows)}", flush=True)
            store[name] = np.array(rows, dtype=float)
            print(f"  {name} done: {store[name].shape}", flush=True)
        np.savez(OUT_NPZ, X_hu=store["human"], X_ai=store["ai"], feature_names=np.array(names))

    # ---- per-feature devC-vs-holdout delta table ----
    Xh, Xa_hold = store["human"], store["ai"]
    dev_c_hu = Xc[yc == 0]
    dev_c_ai = Xc[yc == 1]
    rows = []
    for i, fname in enumerate(names):
        hu_c = dev_c_hu[:, i]
        ai_c = dev_c_ai[:, i]
        hu_c_ok, ai_c_ok = hu_c[np.isfinite(hu_c)], ai_c[np.isfinite(ai_c)]
        roc_c = auroc(list(ai_c_ok), list(hu_c_ok)) if min(len(hu_c_ok), len(ai_c_ok)) >= 50 else np.nan
        orient = 1.0 if (roc_c >= 0.5) else -1.0
        bar = np.nextafter(np.nanmax(hu_c * orient), np.inf)  # C-human max, oriented
        hu_h = Xh[:, i] * orient
        ai_h = Xa_hold[:, i] * orient
        hu_h_ok, ai_h_ok = hu_h[np.isfinite(hu_h)], ai_h[np.isfinite(ai_h)]
        roc_h = auroc(list(ai_h_ok), list(hu_h_ok)) if min(len(hu_h_ok), len(ai_h_ok)) >= 50 else np.nan
        fpr = float(np.mean(hu_h_ok >= bar))
        tpr = float(np.mean(ai_h_ok >= bar))
        rows.append(
            {"feature": fname, "auroc_devC": roc_c, "auroc_holdout": roc_h,
             "delta": (roc_h - roc_c) if np.isfinite(roc_h) and np.isfinite(roc_c) else np.nan,
             "fpr_holdout": fpr, "tpr_holdout": tpr}
        )
    table = pd.DataFrame(rows).sort_values("auroc_holdout", ascending=False, na_position="last")

    # ---- frozen equal-weight ensemble (no learning anywhere) ----
    lo = np.nanpercentile(Xa, 1, axis=0)
    hi = np.nanpercentile(Xa, 99, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    def frozen_scores(X, y_ref):
        out = []
        for i in range(X.shape[1]):
            col = X[:, i]
            hu_c = Xc[yc == 0, i]
            ai_c = Xc[yc == 1, i]
            hu_ok, ai_ok = hu_c[np.isfinite(hu_c)], ai_c[np.isfinite(ai_c)]
            if min(len(hu_ok), len(ai_ok)) < 50:
                continue  # orientation unmeasurable on C: excluded from the mean
            sign = 1.0 if auroc(list(ai_ok), list(hu_ok)) >= 0.5 else -1.0
            out.append(np.clip((col - lo[i]) / span[i], 0, 1) * sign)
        M = np.array(out).T
        return np.nanmean(M, axis=1)

    s_hu = frozen_scores(Xh, None)
    s_ai = frozen_scores(Xa_hold, None)
    c_hu_frozen = frozen_scores(dev_c_hu, None)
    bar = np.nextafter(np.nanmax(c_hu_frozen), np.inf)
    roc = auroc(list(s_ai), list(s_hu))
    fpr = float(np.mean(s_hu >= bar))
    tpr = float(np.mean(s_ai >= bar))
    print(f"\nFROZEN ensemble (no HGB): holdout AUROC {roc:.3f} | FPR {fpr:.5f} | TPR {tpr:.4f}")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("# Holdout breakout: per-feature devC-vs-holdout delta + frozen ensemble\n\n")
        fh.write(f"Frozen equal-weight ensemble (no learning): AUROC {roc:.3f}, "
                 f"FPR {fpr:.5f}, TPR {tpr:.4f}\n\n")
        fh.write("| feature | auroc_devC | auroc_holdout | delta | fpr_holdout | tpr_holdout |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for _, r in table.iterrows():
            fh.write(
                f"| {r.feature} | {r.auroc_devC:.3f} | {r.auroc_holdout:.3f} | "
                f"{r.delta:+.3f} | {r.fpr_holdout:.5f} | {r.tpr_holdout:.4f} |\n"
            )
    print(f"\n{OUT_MD} written")


if __name__ == "__main__":
    main()
