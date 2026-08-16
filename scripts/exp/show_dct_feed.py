"""Same feed dump, original vs char-global-jumbled (exact jumble_response shuffle)."""

import hashlib
import random
import re

from ai_text_detection.shape import RUN, _WORD_RE, dct_run_map

TEXT = "The committee released its findings on Tuesday! Prices rose 4.2% overnight."


def jumbled(text: str) -> str:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rng = random.Random(seed)
    chars = list(text)
    rng.shuffle(chars)
    return "".join(chars)


for label, text in (("ORIGINAL", TEXT), ("JUMBLED", jumbled(TEXT))):
    tokens = _WORD_RE.findall(text)
    lengths = [min(len(w), 15) for w in tokens]
    print(f"===== {label} =====")
    print("text:", text)
    print("tokens:", tokens)
    print("lengths:", lengths)
    print("DCT call inputs (8-runs):")
    for i in range(0, len(lengths) - RUN + 1, RUN):
        print(f"  run at {i}: {lengths[i:i + RUN]}")
    print("symbols out:", list(dct_run_map(text)))
    print()
