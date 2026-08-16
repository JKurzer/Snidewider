"""FLEET F3 v2 — do the classical stats ADD anything over the 89-feat panel?

v1 bug caught: coverage refs pooled bucket A, so A's TRAIN rows measured
against their own grams (no leave-one-out) -> train/serve skew cratered the
panel+full arm (C AUROC 0.917 -> 0.702). v2: exact LOO via reference counters
for A rows. HGB config unified with the head-to-head (tuned params).

Arms (HGB rs=7, train A, read C once per arm; B for reference):
  panel        89 cached features
  panel+stats  + 22 classical (zipf/richness/readability/compress)
  panel+full   + 22 stats + 9 coverage features
  stats+full   the classicals alone (31) — the "no fancy stuff" baseline
Plus: Spearman of the two star coverage contrasts vs existing panel members.

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_stats_stack.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from fleet_coverage import QS, WORD_RE, ngrams
from fleet_stats import FEATS as STAT_FEATS, doc_features
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

OUT = "docs/exp/fleet_stats_stack.md"
COV_NAMES = [f"cov{q}_{s}" for q in QS for s in ("hu", "ai", "contrast")]
# match the head-to-head's tuned config (default HGB is a different yardstick)
HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    # coverage reference COUNTERS from A. Sets lose counts; counters allow
    # exact leave-one-out for A rows (an A doc sits inside its own reference:
    # without LOO its coverage is inflated and the train/serve skew poisons
    # the model - exemplar.py's warning, confirmed by the first run's crater).
    refs: dict[str, dict[int, Counter]] = {"hu": {}, "ai": {}}
    a = buckets["A"]
    for cls, sub in (("hu", a[a.model == "human"]), ("ai", a[a.model != "human"])):
        bags = {q: Counter() for q in QS}
        for t in sub.generation:
            toks = [w.lower() for w in WORD_RE.findall(str(t))]
            for q in QS:
                bags[q].update(ngrams(toks, q))
        refs[cls] = bags

    def cov_vec(text: str, loo: bool) -> list[float]:
        toks = [w.lower() for w in WORD_RE.findall(text)]
        out = []
        for q in QS:
            mine = ngrams(toks, q)
            if not mine:
                out.extend([np.nan] * 3)
                continue
            own = Counter(ngrams(toks, q)) if loo else None
            vals = []
            for cls in ("hu", "ai"):
                refc = refs[cls][q]
                if loo:
                    hits = sum(1 for g in mine if refc.get(g, 0) > own[g])
                else:
                    hits = sum(1 for g in mine if g in refc)
                vals.append(hits / len(mine))
            out.extend([vals[0], vals[1], vals[1] - vals[0]])
        return out

    stats: dict[str, np.ndarray] = {}
    covs: dict[str, np.ndarray] = {}
    for b in "ABC":
        stats[b] = np.array([[doc_features(str(t))[f] for f in STAT_FEATS]
                             for t in buckets[b].generation])
        covs[b] = np.array([cov_vec(str(t), loo=(b == "A"))
                            for t in buckets[b].generation])
        print(f"{b} featurized", flush=True)
    # contamination tell: A-row mean coverage should sit near B/C now
    for q in QS:
        i = COV_NAMES.index(f"cov{q}_contrast")
        print(f"cov{q}_contrast mean: A(loo) {np.nanmean(covs['A'][:, i]):.3f} "
              f"B {np.nanmean(covs['B'][:, i]):.3f} C {np.nanmean(covs['C'][:, i]):.3f}",
              flush=True)

    panel = np.load("data/derived/full_features.npz")
    Xp = {b: panel[f"X_{b}"] for b in "ABC"}
    means = np.nan_to_num(np.nanmean(Xp["A"], axis=0))

    def prep(X):
        X = X.copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(means, bad[1])
        return X

    arms = {
        "panel": lambda b: prep(Xp[b]),
        "panel+stats": lambda b: np.column_stack([prep(Xp[b]), np.nan_to_num(stats[b])]),
        "panel+full": lambda b: np.column_stack([prep(Xp[b]), np.nan_to_num(stats[b]),
                                                 np.nan_to_num(covs[b])]),
        "stats+full": lambda b: np.column_stack([np.nan_to_num(stats[b]),
                                                 np.nan_to_num(covs[b])]),
    }
    lines = ["# FLEET F3 — incremental value of classical stats over the panel\n\n",
             "HGB(rs=7) train A; B reference; C read once per arm. TPR@FPR=1e-2.\n\n",
             "| arm | n_feat | AUROC B | TPR@1e-2 B | AUROC C | TPR@1e-2 C |\n",
             "|---|---|---|---|---|---|\n"]
    for name, get in arms.items():
        Xa = get("A")
        model = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xa, labels["A"])
        row = [name, str(Xa.shape[1])]
        for b in ("B", "C"):
            s = model.predict_proba(get(b))[:, 1]
            roc = auroc(list(s[labels[b] == 1]), list(s[labels[b] == 0]))
            r = tpr_at_fpr(list(s[labels[b] == 1]), list(s[labels[b] == 0]), fpr=1e-2)
            row.extend([f"{roc:.3f}", f"{r['tpr']:.3f} [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}]"])
        lines.append("| " + " | ".join(row) + " |\n")
        print(lines[-1], flush=True)

    # correlation of coverage stars with existing panel members (on C)
    names = list(panel["feature_names"])
    lines.append("\n## Spearman |rho| on C: coverage stars vs panel neighbours\n\n")
    for i, cn in enumerate(COV_NAMES):
        if "contrast" not in cn:
            continue
        rhos = {}
        for j, pn in enumerate(names):
            a_, b_ = covs["C"][:, i], Xp["C"][:, j]
            m = np.isfinite(a_) & np.isfinite(b_)
            if m.sum() > 50:
                ra = np.argsort(np.argsort(a_[m])).astype(float)
                rb = np.argsort(np.argsort(b_[m])).astype(float)
                rhos[pn] = abs(float(np.corrcoef(ra, rb)[0, 1]))
        top = sorted(rhos.items(), key=lambda kv: -kv[1])[:4]
        lines.append(f"- {cn}: " + ", ".join(f"{k} {v:.2f}" for k, v in top) + "\n")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
