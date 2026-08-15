"""Shared dev-fold evaluation plumbing (RULES #4: source-level only).

split_buckets gives three source-disjoint buckets (A/B/C, 50/25/25) with AI
rows capped per source: detectors train on A, meta trains on B, final numbers
on C. Same salt every time — all experiments share the exact same partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AI_PER_SOURCE = 2  # cap AI rows per source; every source contributes its human row


def split_buckets(df: pd.DataFrame, salt: int = 41) -> dict[str, pd.DataFrame]:
    dev = df[df.fold == "dev"]
    sources = np.asarray(dev.source_id.unique())  # numpy array: shuffles for real
    rng = np.random.RandomState(salt)
    rng.shuffle(sources)
    n = len(sources)
    buckets = {
        "A": set(sources[: n // 2]),
        "B": set(sources[n // 2 : 3 * n // 4]),
        "C": set(sources[3 * n // 4 :]),
    }
    out = {}
    for name, ids in buckets.items():
        sub = dev[dev.source_id.isin(ids)]
        humans = sub[sub.model == "human"]
        ai = sub[sub.model != "human"].groupby("source_id").head(AI_PER_SOURCE)
        out[name] = pd.concat([humans, ai])
    return out
