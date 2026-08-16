"""Hanada-search hit features: does the doc CONTAIN near-matching fragments?

Complement to exemplar.py's global profile distance. A bank of byte-chunks
from known-AI and known-human docs (train-half only); per doc we run
qgram.search for each chunk and record hit rates. AI text shares idiomatic
fragments with other AI text; human fragments are more diverse.

Pure per doc given fixed banks (RULES #5). Banks are built once from
bucket-A texts; docs inside a bank must be excluded from detector training
(self-hits are trivially 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_text_detection import qgram

Q = 5
CHUNK = 150  # bytes per exemplar chunk
K = 30  # distance budget for a "hit" (near-exact fragment sharing)

HIT_FEATURE_NAMES = ("hits_ai_rate", "hits_hu_rate", "hits_contrast", "hits_ai_maxrun")


@dataclass
class ChunkBank:
    """Exemplar chunks (middle slice of each bank doc)."""

    chunks: list[bytes]

    @classmethod
    def from_texts(cls, texts: list[str], chunk: int = CHUNK) -> "ChunkBank":
        chunks = []
        for text in texts:
            raw = text.encode("utf-8")
            if len(raw) < chunk:
                continue
            mid = (len(raw) - chunk) // 2
            chunks.append(raw[mid : mid + chunk])
        return cls(chunks)


def hit_features(doc_bytes: bytes, ai_bank: ChunkBank, hu_bank: ChunkBank, k: int = K) -> dict[str, float]:
    """Hit rates of each bank inside the doc + contrast + longest AI hit run."""
    ai_hits = [bool(qgram.search(doc_bytes, chunk, Q, k)) for chunk in ai_bank.chunks]
    hu_hits = [bool(qgram.search(doc_bytes, chunk, Q, k)) for chunk in hu_bank.chunks]
    ai_rate = sum(ai_hits) / len(ai_hits) if ai_hits else 0.0
    hu_rate = sum(hu_hits) / len(hu_hits) if hu_hits else 0.0
    maxrun = 0
    run = 0
    for hit in ai_hits:
        run = run + 1 if hit else 0
        maxrun = max(maxrun, run)
    return {
        "hits_ai_rate": ai_rate,
        "hits_hu_rate": hu_rate,
        "hits_contrast": ai_rate - hu_rate,
        "hits_ai_maxrun": float(maxrun) / len(ai_hits) if ai_hits else 0.0,
    }
