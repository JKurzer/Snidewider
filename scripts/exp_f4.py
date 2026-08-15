"""F4 experiment: exemplar-proximity (doc-vs-corpus) features vs 9-feature baseline.

Protocol mirrors train_tiny.py exactly, so numbers are apples-to-apples with
the 0.898/0.925 (logreg/HGB) biased-long-doc baseline: dev fold only;
humans = all, AI = sample(n=4000, random_state=17); drop NaN-feature docs;
50/50 split by source_id (RandomState(23)). Exemplar banks (200 AI + 200
human, q=3 profiles, computed once) come from the TRAIN half ONLY
(RandomState(101)) — never from test docs; bank members get leave-one-out
distances. If the cost probe projects diff time above DIFF_BUDGET_S, banks
are halved to 100 and re-probed (per F4 brief).

Run: set PYTHONPATH=<worktree>\\src && .venv python scripts/exp_f4.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ai_text_detection
from ai_text_detection import qgram
from ai_text_detection.exemplar import (
    EXEMPLAR_FEATURE_NAMES,
    ExemplarBank,
    exemplar_vector,
)
from ai_text_detection.features import FEATURE_NAMES, feature_vector
from ai_text_detection.metrics import auroc, tpr_at_fpr

PARQUET = Path(r"C:\Users\poly\ai-text-detection\data\derived\raid_splits.parquet")  # read-only
N_AI = 4000
SEED_DATA, SEED_SPLIT, SEED_BANK = 17, 23, 101
N_EXEMPLARS = 200
Q = 3
DIFF_BUDGET_S = 600.0
BASELINE_REF = {"logreg": 0.898, "hgb": 0.925}  # reported biased-subset numbers


def load_docs() -> list[tuple[str, int, str]]:
    df = pd.read_parquet(PARQUET)
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"]
    ai = dev[dev.model != "human"].sample(n=N_AI, random_state=SEED_DATA)
    docs = [(str(t), 0, s) for t, s in zip(humans.generation, humans.source_id)]
    docs += [(str(t), 1, s) for t, s in zip(ai.generation, ai.source_id)]
    return docs


def featurize_baseline(docs):
    X, y, sources, kept = [], [], [], []
    for i, (text, label, source) in enumerate(docs):
        vec = feature_vector(text)
        if any(v != v for v in vec):  # NaN: doc too short for some feature
            continue
        kept.append(i)
        X.append(vec)
        y.append(label)
        sources.append(source)
    return np.array(X), np.array(y), np.array(sources), kept


def split_masks(sources: np.ndarray):
    unique = np.unique(sources)
    rng = np.random.RandomState(SEED_SPLIT)
    train_sources = set(rng.choice(unique, len(unique) // 2, replace=False))
    train_mask = np.array([s in train_sources for s in sources])
    return train_mask, ~train_mask


def build_banks(docs, kept, y, train_mask, max_exemplars):
    """Sample equal-sized exemplar banks from the TRAIN half only, sized
    min(max_exemplars, pools). Returns banks + LOO doc->position maps."""
    rng = np.random.RandomState(SEED_BANK)
    pools = {key: np.where(train_mask & (y == label))[0] for label, key in ((1, "ai"), (0, "hu"))}
    n_ex = min(max_exemplars, *(len(p) for p in pools.values()))
    banks, self_maps = {}, {}
    for key, pool in pools.items():
        chosen = rng.choice(pool, n_ex, replace=False)
        banks[key] = ExemplarBank.from_texts([docs[kept[i]][0] for i in chosen], Q)
        self_maps[key] = {int(doc_i): pos for pos, doc_i in enumerate(chosen)}
    print(f"bank size: {n_ex} per class (pools: ai={len(pools['ai'])}, hu={len(pools['hu'])})")
    return banks["ai"], banks["hu"], self_maps["ai"], self_maps["hu"], n_ex


def probe(docs, kept, ai_bank, hu_bank, n_docs):
    """Measure per-doc profile + diff cost on a mixed subsample. PERF-RULES:
    no numbers, no changes — and no 20-minute surprises either."""
    probe_idx = list(range(0, n_docs, max(1, n_docs // 50)))[:50]
    t0 = time.perf_counter()
    profiles = [qgram.profile(docs[kept[i]][0].encode("utf-8"), Q) for i in probe_idx]
    prof_ms = (time.perf_counter() - t0) / len(probe_idx) * 1000
    t0 = time.perf_counter()
    for p in profiles[:10]:
        exemplar_vector(p, ai_bank, hu_bank)
    diff_ms = (time.perf_counter() - t0) / 10 * 1000
    print(
        f"probe: profile {prof_ms:.1f} ms/doc, bank-diffs {diff_ms:.1f} ms/doc "
        f"-> projected total {(prof_ms + diff_ms) * n_docs / 1000:.0f}s "
        f"({len(ai_bank)}+{len(hu_bank)} exemplars x {n_docs} docs)"
    )
    return prof_ms, diff_ms


def exemplar_matrix(docs, kept, ai_bank, hu_bank, ai_self, hu_self, n_docs):
    X = np.empty((n_docs, len(EXEMPLAR_FEATURE_NAMES)))
    for i in range(n_docs):
        prof = qgram.profile(docs[kept[i]][0].encode("utf-8"), Q)
        X[i] = exemplar_vector(prof, ai_bank, hu_bank, ai_self.get(i), hu_self.get(i))
        if (i + 1) % 1000 == 0:
            print(f"  exemplar features: {i + 1}/{n_docs}")
    return X


def run_models(tag, names, Xtr, ytr, Xte, yte, results):
    for name, model in (
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("hgb", HistGradientBoostingClassifier(random_state=SEED_SPLIT)),
    ):
        model.fit(Xtr, ytr)
        scores = model.predict_proba(Xte)[:, 1]
        roc = auroc(list(scores[yte == 1]), list(scores[yte == 0]))
        res = tpr_at_fpr(list(scores[yte == 1]), list(scores[yte == 0]))
        print(
            f"  {tag:<22} {name:<7} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f}"
            f" [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]"
        )
        results.append((tag, name, roc, res["tpr"], res["tpr_lo"], res["tpr_hi"]))
        if tag.startswith("combined") and name == "logreg":
            coefs = model.named_steps["logisticregression"].coef_[0]
            order = np.argsort(-np.abs(coefs))[:8]
            print("  top combined logreg coefficients (standardized):")
            for i in order:
                print(f"    {names[i]:<22} {coefs[i]:+.3f}")


def main() -> None:
    print(f"ai_text_detection from: {ai_text_detection.__file__}")
    assert "f4-exemplar-search" in ai_text_detection.__file__, "wrong worktree!"
    t_start = time.perf_counter()

    docs = load_docs()
    print(f"docs: {len(docs)} (dev fold; humans + {N_AI} AI sample)")

    t0 = time.perf_counter()
    Xb, y, sources, kept = featurize_baseline(docs)
    print(f"baseline featurize: {time.perf_counter() - t0:.0f}s; usable {len(Xb)} (dropped {len(docs) - len(Xb)} short)")

    train_mask, test_mask = split_masks(sources)
    print(f"train {train_mask.sum()} / test {test_mask.sum()} (source-disjoint)")

    cap = N_EXEMPLARS
    while True:
        t0 = time.perf_counter()
        ai_bank, hu_bank, ai_self, hu_self, n_ex = build_banks(docs, kept, y, train_mask, cap)
        print(f"banks built ({n_ex}+{n_ex} exemplars, train half only): {time.perf_counter() - t0:.1f}s")
        _, diff_ms = probe(docs, kept, ai_bank, hu_bank, len(Xb))
        if diff_ms * len(Xb) / 1000 <= DIFF_BUDGET_S or cap <= 100:
            break
        cap = 100
        print(f"projected diff cost over budget; capping banks at {cap}")

    t0 = time.perf_counter()
    Xe = exemplar_matrix(docs, kept, ai_bank, hu_bank, ai_self, hu_self, len(Xb))
    print(f"exemplar feature matrix: {time.perf_counter() - t0:.0f}s")

    Xb_tr, Xb_te = Xb[train_mask], Xb[test_mask]
    Xe_tr, Xe_te = Xe[train_mask], Xe[test_mask]
    ytr, yte = y[train_mask], y[test_mask]

    print("\nper-feature AUROC (test; sep = max(roc, 1-roc), distances are negatively directed):")
    for i, name in enumerate(EXEMPLAR_FEATURE_NAMES):
        roc = auroc(list(Xe_te[yte == 1, i]), list(Xe_te[yte == 0, i]))
        print(f"  {name:<22} {roc:.3f}  (sep {max(roc, 1 - roc):.3f})")

    results = []
    print("\nmodels (AUROC | TPR@1e-3 Wilson CI):")
    run_models("exemplar-only", EXEMPLAR_FEATURE_NAMES, Xe_tr, ytr, Xe_te, yte, results)
    run_models("baseline-9", FEATURE_NAMES, Xb_tr, ytr, Xb_te, yte, results)
    names_combined = list(FEATURE_NAMES) + list(EXEMPLAR_FEATURE_NAMES)
    run_models(
        "combined-20",
        names_combined,
        np.hstack([Xb_tr, Xe_tr]),
        ytr,
        np.hstack([Xb_te, Xe_te]),
        yte,
        results,
    )

    print("\nsummary (baseline ref: " + ", ".join(f"{k} {v}" for k, v in BASELINE_REF.items()) + "):")
    for tag, name, roc, tpr, lo, hi in results:
        ref = BASELINE_REF.get(name)
        delta = f" ({roc - ref:+.3f} vs ref)" if tag == "baseline-9" else ""
        print(f"  {tag:<22} {name:<7} AUROC {roc:.3f}{delta} | TPR@1e-3 {tpr:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"\ntotal runtime: {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
