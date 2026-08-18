"""FLEET T — jumble-gradient DERIVATIVE features (Donk's shape).

Not CK2 similarity: the RATE at which language-structure collapses as char
jumbling rises. Per doc, at jumble fractions f in {0,.1,.2,.3,.5} (seeded):
  zipf_slope / zipf_r2 (word-frequency rank fit), bg_entropy / cond_entropy
  (char bigram distribution). Raw features = least-squares slope over the
  gradient + the 0 -> 0.5 drop, per metric. 8 features.
Solo bench + increment over the 250 panel.
"""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from jumble_gradient import jumble_fraction  # noqa: E402

from ai_text_detection.evaldata import split_buckets  # noqa: E402
from ai_text_detection.metrics import auroc, tpr_at_fpr  # noqa: E402
from ai_text_detection.stats_features import WORD_RE  # noqa: E402

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_zipf_slope.md"
LEVELS = (0.0, 0.1, 0.2, 0.3, 0.5)
FEATS = ("js_zipf_slope_rate", "js_zipf_slope_drop", "js_zipf_r2_rate",
         "js_zipf_r2_drop", "js_bg_ent_rate", "js_bg_ent_drop",
         "js_cond_ent_rate", "js_cond_ent_drop")


def _zipf(tokens) -> tuple[float, float]:
    counts = np.array(sorted(Counter(tokens).values(), reverse=True))[:50]
    if len(counts) < 10:
        return math.nan, math.nan
    x = np.log(np.arange(1, len(counts) + 1))
    y = np.log(counts)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    resid = y - (ym + slope * (x - xm))
    r2 = float(1 - (resid ** 2).sum() / ((y - ym) ** 2).sum())
    return slope, r2


def _ent(counts, n):
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values())) if n else math.nan


def _char_ents(text: str) -> tuple[float, float]:
    s = text.lower()
    bg = Counter(zip(s, s[1:]))
    n = sum(bg.values())
    h_bg = _ent(bg, n)
    unig = Counter(s)
    h_cond = h_bg - _ent(unig, len(s))
    return h_bg, h_cond


def gradient_metrics(text: str, seed: int) -> np.ndarray:
    rows = []
    for li, f in enumerate(LEVELS):
        jt = jumble_fraction(text, f, seed + li)
        toks = [w.lower() for w in WORD_RE.findall(jt)]
        zs, zr = _zipf(toks)
        hbg, hcond = _char_ents(jt)
        rows.append((zs, zr, hbg, hcond))
    return np.array(rows)  # (n_levels, 4)


def js_features(text: str, seed: int) -> dict[str, float]:
    m = gradient_metrics(text, seed)
    x = np.array(LEVELS)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    out = {}
    names = ("zipf_slope", "zipf_r2", "bg_ent", "cond_ent")
    for j, nm in enumerate(names):
        col = m[:, j]
        if not np.all(np.isfinite(col)):
            out[f"js_{nm}_rate"] = out[f"js_{nm}_drop"] = math.nan
            continue
        rate = float(((x - xm) * (col - col.mean())).sum() / sxx)
        out[f"js_{nm}_rate"] = rate
        out[f"js_{nm}_drop"] = float(col[0] - col[-1])
    return out


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals = {}
    for b in "ABC":
        rows = []
        for i, t in enumerate(buckets[b].generation):
            f = js_features(str(t), seed=10_000 + i)
            rows.append([f[k] for k in FEATS])
        vals[b] = np.array(rows)
        print(f"{b} done", flush=True)

    lines = ["# FLEET T — jumble-gradient derivatives (solo)\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n|---|---|---|---|---|\n"]
    for j, f in enumerate(FEATS):
        rb = eval_feat(vals["B"][:, j], labels["B"])
        rc = eval_feat(vals["C"][:, j], labels["C"])
        lines.append(f"| {f} | {rb[1]:.3f} | {rc[1]:.3f} | {rb[2]:.3f} | {rc[2]:.3f} |\n")

    panel = np.load("data/derived/full_features.npz")
    n_panel = len(panel["feature_names"])
    Xp = {b: panel[f"X_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(Xp["A"], axis=0))

    def prep(X):
        X = X.copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(means, bad[1])
        return X

    new_means = np.nan_to_num(np.nanmean(vals["A"], axis=0))

    def prep_new(b):
        X = vals[b].copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(new_means, bad[1])
        return X

    lines.append("\n## HGB increment (train A, read C)\n\n")
    lines.append("| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |\n|---|---|---|---|---|\n")
    for arm, get in ((f"panel{n_panel}", lambda b: prep(Xp[b])),
                     (f"panel+{n_panel + 8}",
                      lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("js8 alone", lambda b: prep_new(b))):
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
