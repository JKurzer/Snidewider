"""Fragment-hit features: does the doc CONTAIN near-matching bank chunks?

Complement to exemplar.py's global profile distance: local fragment sharing.
Implementation = Ukkonen's counting filter, not the full sliding search
(6000 docs x 300 chunks x sliding windows was ~4B loop iterations; the
filter answers the same question with one gram-index build per doc):

  hit(chunk)  <=>  |codes(chunk) & codes(doc)| / |codes(chunk)| >= tau

Correspondence (Ukkonen 1992): d_q(window, chunk) <= k implies containment
>= 1 - q*k/(|chunk|-q+1). tau=0.7 at q=5, 150B chunks sits just under the
k=8 bound (0.73). qgram.search remains the exact oracle in tests.

Pure per doc given fixed banks (RULES #5). Banks are built once from
bucket-A texts; bank-source docs must be excluded from detector training
(self-hits are trivially 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_text_detection import qgram

Q = 5
CHUNK = 150  # bytes per exemplar chunk
TAU = 0.7  # containment threshold for a "hit" (see module docstring)

HIT_FEATURE_NAMES = ("hits_ai_rate", "hits_hu_rate", "hits_contrast", "hits_ai_maxrun")


@dataclass
class ChunkBank:
    """Exemplar chunks (middle slice of each bank doc) + their gram code sets."""

    chunks: list[bytes]
    code_sets: list[frozenset]

    @classmethod
    def from_texts(cls, texts: list[str], chunk: int = CHUNK, q: int = Q) -> "ChunkBank":
        chunks = []
        code_sets = []
        for text in texts:
            raw = text.encode("utf-8")
            if len(raw) < chunk:
                continue
            mid = (len(raw) - chunk) // 2
            piece = raw[mid : mid + chunk]
            chunks.append(piece)
            code_sets.append(frozenset(code for code, _ in qgram.profile(piece, q)))
        return cls(chunks, code_sets)


def hit_features(
    doc_bytes: bytes, ai_bank: ChunkBank, hu_bank: ChunkBank, tau: float = TAU, q: int = Q
) -> dict[str, float]:
    """Hit rates of each bank inside the doc + contrast + longest AI hit run."""
    doc_codes = {code for code, _ in qgram.profile(doc_bytes, q)}

    def rate(bank: ChunkBank) -> tuple[float, list[bool]]:
        hits = [
            len(cs & doc_codes) / len(cs) >= tau if cs else False
            for cs in bank.code_sets
        ]
        return (sum(hits) / len(hits) if hits else 0.0), hits

    ai_rate, ai_hits = rate(ai_bank)
    hu_rate, _ = rate(hu_bank)
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
