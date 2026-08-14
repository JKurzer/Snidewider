"""One-off/reference PDF -> text extractor for paper ingestion.

Usage: python scripts/extract_pdf.py <input.pdf> <output.txt>

pypdf is in the dev extra. Extraction quality is "good enough for notes" —
two-column layouts may interleave; check against the PDF for quotes.
"""

import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    reader = PdfReader(src)
    pages = [page.extract_text() or "" for page in reader.pages]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(f"<<<{src.name}: {len(pages)} pages>>>\n" + "\n\n".join(pages), encoding="utf-8")
    print(f"{src.name}: {len(pages)} pages -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
