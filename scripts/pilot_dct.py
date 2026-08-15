"""Pilot: do DCT-space features separate human from AI on the dev fold?

Descriptive first look: per-feature AUROC on 1K human + 1K AI dev docs.
Usage: .venv\\Scripts\\python scripts/pilot_dct.py
"""

import pandas as pd

from ai_text_detection import dct
from ai_text_detection.metrics import auroc


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"].sample(n=1000, random_state=31)
    ai = dev[dev.model != "human"].sample(n=1000, random_state=31)

    rows = []
    for label, frame in (("human", humans), ("ai", ai)):
        for text in frame.generation:
            feats = dct.dct_features(str(text))
            feats["label"] = label
            rows.append(feats)
    scores = pd.DataFrame(rows).dropna()
    print(f"usable docs: {len(scores)} / 2000")
    for name in dct.DCT_FEATURE_NAMES:
        h = scores[scores.label == "human"][name]
        a = scores[scores.label == "ai"][name]
        roc = auroc(list(a), list(h))
        print(
            f"  {name:<24} human {h.mean():.3f}  ai {a.mean():.3f}  "
            f"AUROC {roc:.3f}  sep {abs(roc - 0.5) + 0.5:.3f}"
        )


if __name__ == "__main__":
    main()
