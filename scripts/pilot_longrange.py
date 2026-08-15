"""Long-range burst features on the long-doc dev subset only.

Short docs can't support 200-token windows; this measures the long-range
signal where it actually exists. Usage: .venv\\Scripts\\python scripts/pilot_longrange.py
"""

import pandas as pd

from ai_text_detection import burst
from ai_text_detection.metrics import auroc

MIN_TOKENS = 350
CONFIGS = (
    {"window": 100, "gap": -50, "step": 50, "unit": "tokens", "mode": "step"},
    {"window": 100, "gap": 100, "unit": "tokens", "mode": "step"},
    {"window": 200, "gap": 100, "unit": "tokens", "mode": "step"},
    {"window": 150, "samples": 32, "min_gap": 50, "unit": "tokens", "mode": "random"},
    {"window": 200, "samples": 32, "min_gap": 100, "unit": "bytes", "mode": "random"},
)
FEATURES = ("mean", "stdev", "max", "iqr")


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    lens = dev.generation.str.split().str.len()
    for name, mask in (("human", dev.model == "human"), ("ai", dev.model != "human")):
        lengths = lens[mask]
        print(
            f"{name}: total {len(lengths)}, >=350 tokens {int((lengths >= 350).sum())},"
            f" >=500 {int((lengths >= 500).sum())}"
        )

    long_docs = dev[lens >= MIN_TOKENS]
    humans = long_docs[long_docs.model == "human"]
    ai = long_docs[long_docs.model != "human"]
    ai = ai.sample(n=min(len(ai), len(humans) * 4), random_state=13)
    ai_texts = [str(t) for t in ai.generation]
    human_texts = [str(t) for t in humans.generation]
    print(f"measuring on {len(ai_texts)} ai vs {len(human_texts)} human long docs")

    for cfg in CONFIGS:
        label = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"\n-- {label}")
        ai_rows = [burst.burst_features(t, **cfg) for t in ai_texts]
        h_rows = [burst.burst_features(t, **cfg) for t in human_texts]
        for feat in FEATURES:
            pairs = [
                (a[feat], h[feat])
                for a, h in zip(ai_rows, h_rows)
                if a[feat] == a[feat] and h[feat] == h[feat]
            ]
            if not pairs:
                print(f"   {feat:<8} n/a")
                continue
            roc = auroc([a for a, _ in pairs], [h for _, h in pairs])
            print(f"   {feat:<8} AUROC {roc:7.3f}  sep {abs(roc - 0.5) + 0.5:6.3f}   (n={len(pairs)})")


if __name__ == "__main__":
    main()
