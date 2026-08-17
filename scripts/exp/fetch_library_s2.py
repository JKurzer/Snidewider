"""Fallback fetch via Semantic Scholar openAccessPdf links."""
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("papers")
UA = {"User-Agent": "Snidewider-lit-fetch/1.0 (research)"}

TITLES = {
    "liptak-masillo-puglisi2026-matching-statistics-survey":
        "Matching statistics a survey",
    "badkobeh2026-maximal-closed-substrings":
        "Finding maximal closed substrings",
    "alanko2025-finimizers":
        "Finimizers Variable-Length Bounded-Frequency Minimizers for k-mer Sets",
    "diseth2025-parallel-matching-statistics":
        "Massively Parallel Computation of Matching Statistics",
    "alanko2025-spectral-bwt-kmer-lookup":
        "Batched k-Mer Lookup on the Spectral Burrows-Wheeler Transform",
    "rizzo2025-elastic-founder-graphs":
        "Exploiting uniqueness seed-chain-extend alignment on elastic founder graphs",
}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


for name, title in TITLES.items():
    dest = OUT / f"{name}.pdf"
    if dest.exists():
        continue
    try:
        q = urllib.parse.quote(title)
        feed = get("https://api.semanticscholar.org/graph/v1/paper/search"
                   f"?query={q}&limit=3&fields=title,openAccessPdf,externalIds").decode()
        m = re.search(r'"openAccessPdf":\s*{"url":\s*"([^"]+)"', feed)
        if not m:
            print(f"ERR {name}: no OA pdf per S2", flush=True)
            continue
        url = m.group(1).replace("\\u0026", "&")
        dest.write_bytes(get(url))
        print(f"OK  {name} ({dest.stat().st_size // 1024} KB) <- {url[:80]}", flush=True)
    except Exception as exc:
        print(f"ERR {name}: {exc}", flush=True)
    time.sleep(8)

print("done")
