"""F2 length-control: how much of the 9-feature pilot is real vs length artifact?

Suspected confound: qgram_total == len(text_bytes) - q + 1, i.e. document
length wearing a trenchcoat, and RAID dev AI docs are systematically shorter
(zero AI docs >=500 tokens vs 195 human).

Configs (dev fold only; humans 2000 + AI 4000 rs=17; 50/50 source split rs=23):
  A  full 9 features, logreg + HGB            -> replicate pilot ~0.898/0.925
  B  ablate qgram_total (length-pure feature) -> re-measure
  C  length-matched eval: token bins with both classes, class-balanced per bin
     C1 models trained on full train, evaluated on matched test
     C2 models trained AND evaluated on length-matched subsets

Usage: set PYTHONPATH=src && python scripts/exp_f2.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ai_text_detection
from ai_text_detection.features import FEATURE_NAMES, feature_vector
from ai_text_detection.metrics import auroc, tpr_at_fpr

DATA = r"C:\Users\poly\ai-text-detection\data\derived\raid_splits.parquet"
N_AI = 4000
# Featurizable docs need >=350 tokens (midrange: 2*150+50). Bins chosen so both
# classes are present (AI has zero docs >=500 tokens).
BINS = [(350, 375), (375, 400), (400, 450), (450, 500)]
BALANCE_SEED = 42

DROP_FOR_ABLATION = ["qgram_total"]  # length-pure: total == n_bytes - q + 1


def load_docs() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"]
    ai = dev[dev.model != "human"].sample(n=N_AI, random_state=17)
    docs = pd.concat([humans, ai])
    return pd.DataFrame(
        {
            "text": docs.generation.astype(str),
            "label": (docs.model != "human").astype(int),
            "source": docs.source_id,
            "ntok": docs.generation.astype(str).str.split().str.len(),
        }
    ).reset_index(drop=True)


def featurize(docs: pd.DataFrame):
    vecs, keep = [], []
    for i, text in enumerate(docs.text):
        vec = feature_vector(text)
        if any(v != v for v in vec):  # NaN: too short for some window
            continue
        vecs.append(vec)
        keep.append(i)
    X = np.array(vecs)
    return X, docs.iloc[keep].reset_index(drop=True)


def source_split(sources: pd.Series, seed: int = 23) -> np.ndarray:
    unique = np.unique(sources)
    rng = np.random.RandomState(seed)
    train_sources = set(rng.choice(unique, len(unique) // 2, replace=False))
    return np.array([s in train_sources for s in sources])


def make_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(random_state=23),
    }


def evaluate(model, Xtr, ytr, Xte, yte) -> dict:
    model.fit(Xtr, ytr)
    scores = model.predict_proba(Xte)[:, 1]
    res = tpr_at_fpr(list(scores[yte == 1]), list(scores[yte == 0]))
    return {
        "auroc": auroc(list(scores[yte == 1]), list(scores[yte == 0])),
        "tpr": res["tpr"],
        "tpr_lo": res["tpr_lo"],
        "tpr_hi": res["tpr_hi"],
        "scores": scores,
    }


def report(tag: str, r: dict, n_te: int) -> None:
    print(
        f"  {tag:<10} AUROC {r['auroc']:.3f} | TPR@1e-3 {r['tpr']:.3f}"
        f" [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}] (n_test={n_te})"
    )


def length_matched(docs: pd.DataFrame, seed: int = BALANCE_SEED) -> np.ndarray:
    """Indices of a bin-class-balanced subsample: per bin, min(n_ai, n_human)
    of each class -> identical length marginals for both classes."""
    rng = np.random.RandomState(seed)
    picked = []
    for lo, hi in BINS:
        in_bin = np.where((docs.ntok >= lo) & (docs.ntok < hi))[0]
        per_class = [in_bin[docs.label.iloc[in_bin].to_numpy() == c] for c in (0, 1)]
        n = min(len(per_class[0]), len(per_class[1]))
        if n == 0:
            continue
        for cls_idx in per_class:
            picked.extend(rng.choice(cls_idx, n, replace=False))
    return np.sort(picked)


def main() -> None:
    print(f"ai_text_detection: {ai_text_detection.__file__}")
    docs = load_docs()
    n_raw = len(docs)
    X, docs = featurize(docs)
    y = docs.label.to_numpy()
    print(f"usable docs: {len(X)} (dropped {n_raw - len(X)} short; need >=350 tokens)")

    # --- confound check: length table + per-feature length coupling (test) ---
    print("\ntoken-count distribution of usable docs:")
    tab = pd.crosstab(
        pd.cut(docs.ntok, [b[0] for b in BINS] + [500, 10**9], right=False),
        docs.label.map({0: "human", 1: "ai"}),
    )
    print(tab.to_string())

    train_mask = source_split(docs.source)
    te = ~train_mask
    Xtr, Xte, ytr, yte = X[train_mask], X[te], y[train_mask], y[te]
    ntok_te = docs.ntok.to_numpy()[te]
    print(f"\ntrain {len(Xtr)} / test {len(Xte)} (source-disjoint)")

    print("\nper-feature: AUROC (test) | Spearman corr with token count (test)")
    for i, name in enumerate(FEATURE_NAMES):
        roc = auroc(list(Xte[yte == 1, i]), list(Xte[yte == 0, i]))
        rho = spearmanr(Xte[:, i], ntok_te).statistic
        print(f"  {name:<22} {roc:.3f} | rho={rho:+.3f}")
    roc_len = auroc(list(ntok_te[yte == 1]), list(ntok_te[yte == 0]))
    print(f"  {'LENGTH-ONLY (ntok)':<22} {roc_len:.3f} | rho=+1.000")

    # --- config A: full features (replicate pilot) ---
    print("\n[A] full 9 features, biased long-doc subset:")
    res_a = {name: evaluate(m, Xtr, ytr, Xte, yte) for name, m in make_models().items()}
    for name, r in res_a.items():
        report(name, r, len(Xte))

    # --- config B: ablate qgram_total ---
    keep_idx = [i for i, n in enumerate(FEATURE_NAMES) if n not in DROP_FOR_ABLATION]
    print(f"\n[B] ablated ({len(keep_idx)} features, dropped {DROP_FOR_ABLATION}):")
    res_b = {
        name: evaluate(m, Xtr[:, keep_idx], ytr, Xte[:, keep_idx], yte)
        for name, m in make_models().items()
    }
    for name, r in res_b.items():
        report(name, r, len(Xte))

    # --- config C: length-matched eval ---
    matched_te = length_matched(docs[te])
    matched_tr = length_matched(docs[train_mask])
    print(
        f"\n[C] length-matched: bins {BINS}; matched train={len(matched_tr)}"
        f" test={len(matched_te)} (class-balanced per bin)"
    )
    Xm, ym = Xte[matched_te], yte[matched_te]
    print("  C1: full-trained models, matched test:")
    for name, r in res_a.items():
        s = r["scores"][matched_te]
        res = tpr_at_fpr(list(s[ym == 1]), list(s[ym == 0]))
        roc = auroc(list(s[ym == 1]), list(s[ym == 0]))
        report(name, {"auroc": roc, **res}, len(Xm))
    print("  C2: retrained on matched train, matched test (full features):")
    Xtr_m, ytr_m = X[train_mask][matched_tr], ytr[matched_tr]
    res_c2 = {n: evaluate(m, Xtr_m, ytr_m, Xm, ym) for n, m in make_models().items()}
    for name, r in res_c2.items():
        report(name, r, len(Xm))
    print("  C2-ablated: retrained on matched train, matched test, no qgram_total:")
    res_c2b = {
        n: evaluate(m, Xtr_m[:, keep_idx], ytr_m, Xm[:, keep_idx], ym)
        for n, m in make_models().items()
    }
    for name, r in res_c2b.items():
        report(name, r, len(Xm))

    # --- per-bin separation (all test docs in bin, full + ablated models) ---
    print("\nper-bin AUROC on test (full-trained models; n_human/n_ai per bin):")
    docs_te = docs[te]
    for lo, hi in BINS:
        in_bin = np.where((docs_te.ntok >= lo) & (docs_te.ntok < hi))[0]
        yb = yte[in_bin]
        if (yb == 0).sum() == 0 or (yb == 1).sum() == 0:
            continue
        cells = []
        for name in ("logreg", "hgb"):
            sa = res_a[name]["scores"][in_bin]
            sb = res_b[name]["scores"][in_bin]
            cells.append(
                f"{name}: full={auroc(list(sa[yb == 1]), list(sa[yb == 0])):.3f}"
                f" abl={auroc(list(sb[yb == 1]), list(sb[yb == 0])):.3f}"
            )
        print(f"  {lo}-{hi}: n_h={(yb == 0).sum():<4} n_ai={(yb == 1).sum():<4} " + " | ".join(cells))


if __name__ == "__main__":
    main()
