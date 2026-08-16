"""Bench the gremlin: char-global-jumble response as a feature, no theories.

feature(doc) = CK2(dct_run_map(doc) || dct_run_map(char-global-jumble(doc)))
Seeded per doc (pure function). Benched like any feature: direction-corrected
AUROC, zero-FPR TPR, TPR@1e-3 with achieved FPR + Wilson CI, coverage.
Usage: .venv\\Scripts\\python scripts/exp/jumble_response.py
"""

import hashlib
import random

import numpy as np
import pandas as pd

from ai_text_detection import ck2
from ai_text_detection.metrics import auroc, tpr_at_fpr, zero_fpr_tpr
from ai_text_detection.shape import dct_run_map


def jumble_response(text: str) -> float:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rng = random.Random(seed)
    chars = list(text)
    rng.shuffle(chars)
    orig = dct_run_map(text)
    if len(orig) < 4:
        return float("nan")
    return ck2.similarity(orig, dct_run_map("".join(chars)))


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"].sample(n=1000, random_state=71)
    ai = dev[dev.model != "human"].sample(n=1000, random_state=71)

    rows = []
    for label, frame in (("human", humans), ("ai", ai)):
        for text in frame.generation:
            rows.append((label, jumble_response(str(text))))
    scores = pd.DataFrame(rows, columns=["label", "score"]).dropna()
    print(f"coverage: {len(scores)}/2000 docs")

    h = scores[scores.label == "human"].score
    a = scores[scores.label == "ai"].score
    print(f"human mean {h.mean():.4f} sd {h.std():.4f} | ai mean {a.mean():.4f} sd {a.std():.4f}")

    roc = auroc(list(a), list(h))
    oriented_a, oriented_h = (list(a), list(h)) if roc >= 0.5 else (list(-a), list(-h))
    res = tpr_at_fpr(oriented_a, oriented_h)
    gate = zero_fpr_tpr(oriented_a, oriented_h)
    print(f"AUROC {roc:.3f} (sep {abs(roc - 0.5) + 0.5:.3f})")
    print(
        f"TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}] "
        f"achieved-FPR {res['fpr_achieved']:.4f}"
    )
    print(
        f"zero-FPR gate: TPR {gate['tpr']:.3f} [{gate['tpr_lo']:.3f}, {gate['tpr_hi']:.3f}] "
        f"(n_h={int(gate['n_human'])})"
    )


if __name__ == "__main__":
    main()
