"""Pull the adversarial training supplement from the RAID train split.

Per Donk: 20K human + 20K AI rows per attack. HYGIENE (RULES #4): only rows
whose source_id is OUTSIDE our dev+holdout folds (an attacked variant of a
C/holdout generation in training would be a same-document leak; the ~430K
train rows outside our folds are clean game).

Output: data/derived/adv_supplement.parquet
Usage: .venv\\Scripts\\python scripts\\pull_adv_supplement.py
"""

import pandas as pd
from raid.utils import load_data

PER_CLASS = 20_000
OUT = "data/derived/adv_supplement.parquet"


def main() -> None:
    ours = pd.read_parquet("data/derived/raid_splits.parquet")
    a_sources = set(ours[ours.bucket == "A"].source_id) if "bucket" in ours else \
        set(__import__("ai_text_detection.evaldata", fromlist=["split_buckets"])
            .split_buckets(ours)["A"].source_id)
    print(f"bucket-A sources: {len(a_sources)}", flush=True)

    train = load_data(split="train", include_adversarial=True)
    print(f"train rows: {len(train)}", flush=True)
    adv = train[(train.attack != "none") & (train.source_id.isin(a_sources))]
    print(f"adversarial rows from A sources: {len(adv)}", flush=True)

    parts = []
    for attack, sub in adv.groupby("attack"):
        hu = sub[sub.model == "human"]
        ai = sub[sub.model != "human"]
        take_hu = hu.sample(min(PER_CLASS, len(hu)), random_state=7)
        take_ai = ai.sample(min(PER_CLASS, len(ai)), random_state=7)
        parts.append(pd.concat([take_hu, take_ai]))
        print(f"  {attack}: hu {len(take_hu)} ai {len(take_ai)}", flush=True)

    supp = pd.concat(parts)[["id", "source_id", "model", "attack", "domain",
                             "generation"]].reset_index(drop=True)
    supp.to_parquet(OUT)
    print(f"{OUT}: {len(supp)} rows", flush=True)


if __name__ == "__main__":
    main()
