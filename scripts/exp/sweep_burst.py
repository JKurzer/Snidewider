"""Sweep burst window/stride configs on the RAID dev sample.

Rank-statistic AUROC (Mann-Whitney) per config: direction-agnostic separation
measure, no threshold committed. Top configs then get calibrated TPR@FPR~1e-3
on the full 2K+2K sample. Usage: .venv\\Scripts\\python scripts/sweep_burst.py
"""

import pandas as pd

from ai_text_detection import burst
from ai_text_detection.metrics import auroc

WINDOWS = (5, 10, 20, 40, 80)
BYTE_CONFIGS = ((200, 100), (200, 50), (150, 100), (250, 125), (254, 254))


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"].sample(n=1000, random_state=11)
    ai = dev[dev.model != "human"].sample(n=1000, random_state=11)

    human_texts = [str(t) for t in humans.generation]
    ai_texts = [str(t) for t in ai.generation]

    print(f"sweep on {len(ai_texts)} ai vs {len(human_texts)} human (ck2 metric)")
    print(f"{'window':>6} {'stride':>6} {'AUROC':>7} {'sep':>6}")
    results = []
    for window in WINDOWS:
        for stride in sorted({1, max(1, window // 4), max(1, window // 2), window}):
            ai_means = [
                burst.burst_features(t, window, gap=stride - window, step=stride)["mean"]
                for t in ai_texts
            ]
            h_means = [
                burst.burst_features(t, window, gap=stride - window, step=stride)["mean"]
                for t in human_texts
            ]
            pairs = [(a, h) for a, h in zip(ai_means, h_means) if a == a and h == h]
            roc = auroc([a for a, _ in pairs], [h for _, h in pairs])
            results.append((roc, window, stride))
            print(f"{window:>6} {stride:>6} {roc:7.3f} {abs(roc - 0.5) + 0.5:6.3f}")

    print("\n== long-range configs (per-feature AUROC) ==")
    lr_configs = (
        {"window": 200, "gap": -100, "step": 100, "unit": "tokens", "mode": "step"},
        {"window": 200, "gap": 100, "unit": "tokens", "mode": "step"},
        {"window": 200, "samples": 32, "min_gap": 100, "unit": "bytes", "mode": "random"},
        {"window": 200, "samples": 64, "min_gap": 200, "unit": "bytes", "mode": "random"},
        {"window": 150, "samples": 32, "min_gap": 50, "unit": "tokens", "mode": "random"},
    )
    feats = ("mean", "stdev", "max", "iqr")
    for cfg in lr_configs:
        label = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"\n-- {label}")
        ai_rows = [burst.burst_features(t, **cfg) for t in ai_texts]
        h_rows = [burst.burst_features(t, **cfg) for t in human_texts]
        for feat in feats:
            pairs = [
                (a[feat], h[feat])
                for a, h in zip(ai_rows, h_rows)
                if a[feat] == a[feat] and h[feat] == h[feat]
            ]
            if not pairs:
                print(f"   {feat:<8} n/a (no docs long enough)")
                continue
            roc = auroc([a for a, _ in pairs], [h for _, h in pairs])
            print(f"   {feat:<8} AUROC {roc:7.3f}  sep {abs(roc - 0.5) + 0.5:6.3f}   (n={len(pairs)})")


if __name__ == "__main__":
    main()
