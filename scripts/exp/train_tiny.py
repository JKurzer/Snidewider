"""Tiny classifier pilot: 9 dumb features + logistic regression / HGB.

Dev fold only, split 50/50 by source_id. Reports AUROC, TPR@FPR=1e-3 with
Wilson CI, and per-feature AUROC + logistic coefficients (which family
carries). Usage: .venv\\Scripts\\python scripts/train_tiny.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_text_detection.features import FEATURE_NAMES, feature_vector
from ai_text_detection.metrics import auroc, tpr_at_fpr

N_AI = 4000


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"]
    ai = dev[dev.model != "human"].sample(n=N_AI, random_state=17)

    docs = [(t, 0, s) for t, s in zip(humans.generation, humans.source_id)]
    docs += [(t, 1, s) for t, s in zip(ai.generation, ai.source_id)]

    print(f"featurizing {len(docs)} docs...")
    X, y, sources = [], [], []
    dropped = 0
    for text, label, source in docs:
        vec = feature_vector(str(text))
        if any(v != v for v in vec):  # NaN: doc too short for some feature
            dropped += 1
            continue
        X.append(vec)
        y.append(label)
        sources.append(source)
    X, y = np.array(X), np.array(y)
    sources = np.array(sources)
    print(f"usable docs: {len(X)} (dropped {dropped} short docs)")

    unique_sources = np.unique(sources)
    rng = np.random.RandomState(23)
    train_sources = set(rng.choice(unique_sources, len(unique_sources) // 2, replace=False))
    train_mask = np.array([s in train_sources for s in sources])
    Xtr, Xte, ytr, yte = X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]
    print(f"train {len(Xtr)} / test {len(Xte)} (source-disjoint)")

    print("\nper-feature AUROC (test):")
    for i, name in enumerate(FEATURE_NAMES):
        roc = auroc(list(Xte[yte == 1, i]), list(Xte[yte == 0, i]))
        print(f"  {name:<22} {roc:.3f}")

    for name, model in (
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("hgb", HistGradientBoostingClassifier(random_state=23)),
    ):
        model.fit(Xtr, ytr)
        scores = model.predict_proba(Xte)[:, 1]
        roc = auroc(list(scores[yte == 1]), list(scores[yte == 0]))
        res = tpr_at_fpr(list(scores[yte == 1]), list(scores[yte == 0]))
        print(
            f"\n{name}: AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f}"
            f" [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}] (n_ai={int(res['n_ai'])})"
        )
        if name == "logreg":
            coefs = model.named_steps["logisticregression"].coef_[0]
            order = np.argsort(-np.abs(coefs))
            print("  coefficients (standardized):")
            for i in order:
                print(f"    {FEATURE_NAMES[i]:<22} {coefs[i]:+.3f}")


if __name__ == "__main__":
    main()
