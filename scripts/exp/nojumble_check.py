"""Bug-or-strange: the EXACT jumble_response code path with jumble disabled.

no-jumble feature = CK2(dct_run_map(text), dct_run_map(text)) via the identical
pipeline (same map calls, same ck2 call). Expected: exactly 0.0 for every doc
(identity). Any nonzero value = a bug in the path (nondeterminism).
Compares against the vanilla jumble-response numbers side by side.
Usage: .venv\\Scripts\\python scripts/exp/nojumble_check.py
"""

import hashlib
import random

import pandas as pd

from ai_text_detection import ck2
from ai_text_detection.metrics import auroc
from ai_text_detection.shape import dct_run_map


def response(text: str, jumble: bool) -> float:
    """The exact jumble_response code path; jumble=False disables the shuffle."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rng = random.Random(seed)
    chars = list(text)
    if jumble:
        rng.shuffle(chars)  # ONLY this step is disabled in the no-jumble arm
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
            text = str(text)
            rows.append((label, response(text, jumble=False), response(text, jumble=True)))
    scores = pd.DataFrame(rows, columns=["label", "nojumble", "jumble"]).dropna()

    nj = scores.nojumble
    print(f"docs: {len(scores)}")
    print(f"no-jumble: min {nj.min()} max {nj.max()} mean {nj.mean()} — nonzero count: {(nj != 0).sum()}")
    for label, sub in scores.groupby("label"):
        print(f"{label:>6}: jumble mean {sub.jumble.mean():.4f} | no-jumble mean {sub.nojumble.mean():.4f}")
    roc = auroc(
        list(scores[scores.label == 'ai'].jumble), list(scores[scores.label == 'human'].jumble)
    )
    print(f"jumble-response AUROC this sample: {roc:.3f}")


if __name__ == "__main__":
    main()
