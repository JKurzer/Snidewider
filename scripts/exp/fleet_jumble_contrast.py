"""FLEET V — jumble CONTRAST features (Donk: 'there's something in there').

Not "how much survives" (fleet C/T) but "how much was positionally loaded":
  feature = stat(original) - stat(jumbled)
AI recurrence is positional (anti-repeat glue clusters repeats just outside
the detection window); human recurrence is lexical (words reused wherever).
So the drop under a partial shuffle should separate the classes.

4 raw features:
  oct_order_load   oct_hits drop under 30% token shuffle
  reuse_order_load peak_reuse_abs drop under 30% token shuffle
  zipf_struct_load zipf R^2 drop under 30% char jumble
  cond_order_load  char conditional-entropy drop under 30% char jumble

Solo bench + increment over the 250 panel.
"""
import math
import random
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
from ai_text_detection.token_bigrams import oct_hits_features, token_reuse_features  # noqa: E402

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_jumble_contrast.md"
FEATS = ("oct_order_load", "reuse_order_load", "zipf_struct_load", "cond_order_load")


def token_jumble(text: str, frac: float, seed: int) -> str:
    rng = random.Random(seed)
    toks = WORD_RE.findall(text)
    n = len(toks)
    idx = list(range(n))
    k = int(n * frac)
    chosen = rng.sample(idx, k)
    vals = [toks[i] for i in chosen]
    rng.shuffle(vals)
    for i, v in zip(chosen, vals):
        toks[i] = v
    return " ".join(toks)


def _zipf_r2(tokens) -> float:
    counts = np.array(sorted(Counter(tokens).values(), reverse=True))[:50]
    if len(counts) < 10:
        return math.nan
    x = np.log(np.arange(1, len(counts) + 1))
    y = np.log(counts)
    xm, ym = x.mean(), y.mean()
    slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    resid = y - (ym + slope * (x - xm))
    denom = ((y - ym) ** 2).sum()
    return float(1 - (resid ** 2).sum() / denom) if denom else math.nan


def _cond_ent(text: str) -> float:
    s = text.lower()
    bg = Counter(zip(s, s[1:]))
    n = sum(bg.values())
    if not n:
        return math.nan
    h_bg = -sum((c / n) * math.log2(c / n) for c in bg.values())
    unig = Counter(s)
    h_uni = -sum((c / len(s)) * math.log2(c / len(s)) for c in unig.values())
    return h_bg - h_uni


def jc_features(text: str, seed: int) -> dict[str, float]:
    toks_orig = [w.lower() for w in WORD_RE.findall(text)]
    tj = token_jumble(text, 0.3, seed)
    cj = jumble_fraction(text, 0.3, seed + 7)
    return {
        "oct_order_load": (oct_hits_features(text)["oct_hits"]
                           - oct_hits_features(tj)["oct_hits"]),
        "reuse_order_load": (token_reuse_features(text)["peak_reuse_abs"]
                             - token_reuse_features(tj)["peak_reuse_abs"]),
        "zipf_struct_load": _zipf_r2(toks_orig)
                            - _zipf_r2([w.lower() for w in WORD_RE.findall(cj)]),
        "cond_order_load": _cond_ent(text) - _cond_ent(cj),
    }


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals = {}
    for b in "ABC":
        vals[b] = np.array([[jc_features(str(t), 20_000 + i)[k] for k in FEATS]
                            for i, t in enumerate(buckets[b].generation)])
        print(f"{b} done", flush=True)

    lines = ["# FLEET V — jumble contrast (positional loading) solo\n\n",
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
                     (f"panel+{n_panel + 4}",
                      lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("jc4 alone", lambda b: prep_new(b))):
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
