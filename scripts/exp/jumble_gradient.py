"""Jumble-gradient probe: 10 passes of increasing global char-jumble per doc.

feature(doc) = [CK2(dct_run_map(text), dct_run_map(jumble_f(text))) for f in 0.1..1.0]
Partial jumble = shuffle a uniformly sampled fraction f of character positions
(seeded per doc per level: pure function, RULES #5).

Bench: per-level AUROC + a small HGB on the 10-vector. Donk is hoping for
nothing. Usage: .venv\\Scripts\\python scripts/exp/jumble_gradient.py
"""

import hashlib
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import ck2
from ai_text_detection.metrics import auroc, tpr_at_fpr
from ai_text_detection.shape import dct_run_map

LEVELS = [f / 10 for f in range(1, 11)]  # 0.1 .. 1.0


def jumble_fraction(text: str, frac: float, seed: int) -> str:
    rng = random.Random(seed)
    chars = list(text)
    k = int(frac * len(chars))
    idx = rng.sample(range(len(chars)), k)
    sub = [chars[i] for i in idx]
    rng.shuffle(sub)
    for i, c in zip(idx, sub):
        chars[i] = c
    return "".join(chars)


def gradient(text: str) -> list[float]:
    orig = dct_run_map(text)
    if len(orig) < 4:
        return [float("nan")] * len(LEVELS)
    base_seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    return [
        ck2.similarity(orig, dct_run_map(jumble_fraction(text, f, base_seed + li)))
        for li, f in enumerate(LEVELS)
    ]


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"].sample(n=1000, random_state=83)
    ai = dev[dev.model != "human"].sample(n=1000, random_state=83)

    rows, labels = [], []
    for label, frame in ((0, humans), (1, ai)):
        for text in frame.generation:
            rows.append(gradient(str(text)))
            labels.append(label)
    X = np.array(rows)
    y = np.array(labels)
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    print(f"coverage: {len(X)}/2000 docs")

    print("\nper-level AUROC:")
    for li, f in enumerate(LEVELS):
        roc = auroc(list(X[y == 1, li]), list(X[y == 0, li]))
        print(f"  f={f:.1f}  AUROC {roc:.3f}  (human {X[y==0, li].mean():.3f} vs ai {X[y==1, li].mean():.3f})")

    idx = np.arange(len(X))
    rng = np.random.RandomState(5)
    rng.shuffle(idx)
    half = len(idx) // 2
    tr, te = idx[:half], idx[half:]
    model = HistGradientBoostingClassifier(random_state=7).fit(X[tr], y[tr])
    s = model.predict_proba(X[te])[:, 1]
    roc = auroc(list(s[y[te] == 1]), list(s[y[te] == 0]))
    res = tpr_at_fpr(list(s[y[te] == 1]), list(s[y[te] == 0]))
    print(f"\n10-vector HGB (held-out half): AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")
    in_s = model.predict_proba(X[tr])[:, 1]
    print(f"(in-sample on train half, for scale: AUROC {auroc(list(in_s[y[tr]==1]), list(in_s[y[tr]==0])):.3f})")


if __name__ == "__main__":
    main()
