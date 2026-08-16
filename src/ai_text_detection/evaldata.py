"""Shared dev-fold evaluation plumbing (RULES #4: source-level only).

split_buckets gives three source-disjoint buckets (A/B/C, 50/25/25) with AI
rows capped per source: detectors train on A, meta trains on B, final numbers
on C. Same salt every time — all experiments share the exact same partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AI_PER_SOURCE = 2  # cap AI rows per source; every source contributes its human row


def _ai_sample(ai: pd.DataFrame, k: int, salt: int) -> pd.DataFrame:
    """Seeded uniform sample of k AI rows per source.

    The parquet stores each source's generations MODEL-ORDERED, so
    groupby().head(k) silently takes the same model (llama-chat) for every
    source — FLEET D audit (docs/exp/fleet_holdout_audit.md): head(2) made
    dev AI 100% llama-chat while holdout is 11 models, fabricating a uniform
    ~2.8x dev->holdout sampling tax. Uniform slot sampling matches the
    holdout mix in expectation.
    """
    rng = np.random.RandomState(salt)
    parts = []
    for _, group in ai.groupby("source_id"):  # groupby sorts source_ids: deterministic
        n = len(group)
        idx = rng.choice(n, size=min(k, n), replace=False)
        parts.append(group.iloc[sorted(idx)])
    return pd.concat(parts)


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
        ai = _ai_sample(sub[sub.model != "human"], AI_PER_SOURCE, salt)
        out[name] = pd.concat([humans, ai])
    return out
