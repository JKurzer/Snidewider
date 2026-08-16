"""Decompose G5's Probe 3 number: CK2(orig || global-char-jumble) = 0.814.

CK2 score = 1 - percentOrdered * percentSimilarChars. Token-jumble permutes
the same symbol multiset (chars ~1.0); global char jumble changes BOTH order
and the multiset. This decomposes both into their order/char components so
we can see exactly why 0.697 vs 0.814 — and why 0.814 isn't ~1.0.
"""

import random

import pandas as pd

from ai_text_detection import ck2
from ai_text_detection.shape import dct_run_map

rng = random.Random(99)

df = pd.read_parquet("data/derived/raid_splits.parquet")
docs = df[df.fold == "dev"].generation.sample(300, random_state=99)


def jumble_tokens(text: str) -> str:
    toks = text.split()
    rng.shuffle(toks)
    return " ".join(toks)


def jumble_chars_global(text: str) -> str:
    chars = list(text)
    rng.shuffle(chars)
    return "".join(chars)


rows = []
for text in docs:
    text = str(text)
    orig = dct_run_map(text)
    if len(orig) < 4:
        continue
    for label, jumbled in (("token-jumble", jumble_tokens(text)), ("char-global", jumble_chars_global(text))):
        stream = dct_run_map(jumbled)
        m = ck2._ck2_native.measures(orig, stream)
        n = m["n"]
        rows.append(
            {
                "probe": label,
                "score": m["score"],
                "pct_ordered": (n * n - m["D"]) / (n * n),
                "pct_similar_chars": (n - m["S"]) / n,
            }
        )

out = pd.DataFrame(rows)
print(out.groupby("probe")[["score", "pct_ordered", "pct_similar_chars"]].mean().round(3))
