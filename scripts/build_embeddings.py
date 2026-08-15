"""Build the compact embedding store for the DCT encoder.

Parses the raw GloVe 300d file (data/raw/embeddings/glove.6B.300d.txt,
immutable), keeps only tokens that appear in our corpora (dev fold + selfgen),
caps at --max-vocab by GloVe order (roughly frequency order), and writes
data/derived/embeddings.npz (vocab + float32 matrix).

Usage: .venv\\Scripts\\python scripts/build_embeddings.py [--max-vocab 60000]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GLOVE = Path("data/raw/embeddings/glove.6B.300d.txt")
OUT = Path("data/derived/embeddings.npz")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def corpus_tokens() -> set[str]:
    tokens: set[str] = set()
    df = pd.read_parquet("data/derived/raid_splits.parquet", columns=["generation"])
    for text in df.generation:
        tokens.update(TOKEN_RE.findall(str(text).lower()))
    for f in Path("data/raw/selfgen").rglob("*.txt"):
        tokens.update(TOKEN_RE.findall(f.read_text(encoding="utf-8").lower()))
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-vocab", type=int, default=60_000)
    args = parser.parse_args()

    wanted = corpus_tokens()
    print(f"corpus tokens: {len(wanted)}; filtering GloVe...")

    vocab: list[str] = []
    vectors: list[np.ndarray] = []
    with GLOVE.open(encoding="utf-8") as fh:
        for line in fh:
            if len(vocab) >= args.max_vocab:
                break
            word, _, rest = line.rstrip().partition(" ")
            if word in wanted:
                vocab.append(word)
                vectors.append(np.fromstring(rest, sep=" ", dtype=np.float32))
    matrix = np.vstack(vectors)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, vocab=np.array(vocab), matrix=matrix)
    print(f"{OUT}: {matrix.shape} ({OUT.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
