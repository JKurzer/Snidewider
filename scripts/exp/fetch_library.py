"""Pull the Helsinki-school library into papers/lit/ (arXiv + open-access)."""
import re
import time
import urllib.request
from pathlib import Path

OUT = Path("papers")
OUT.mkdir(parents=True, exist_ok=True)

ARXIV = {
    "puglisi2026-repetitiveness-reversal": "https://arxiv.org/pdf/2602.14385",
    "puglisi2025-compressed-dict-matching-rle": "https://arxiv.org/pdf/2509.03265",
    "makinen2026-adaptive-compressed-suffix-arrays": "https://arxiv.org/pdf/2602.17201",
    "makinen2026-quantum-pattern-matching-degenerate": "https://arxiv.org/pdf/2603.16297",
    "makinen2025-practical-colinear-chaining": "https://arxiv.org/pdf/2506.11750",
}
# open-access drops.dagstuhl.de documents (LIPIcs/OASIcs)
DROPS = {
    "donges-puglisi2025-succinct-rank-dictionaries": "https://doi.org/10.4230/LIPIcs.SEA.2025.15",
    "alanko2025-graph-indexing-beyond-wheeler": "https://doi.org/10.4230/OASIcs.Manzini.13",
    "makinen2025-elastic-degenerate-strings-construction": "https://doi.org/10.4230/OASIcs.Grossi.2",
    "alanko2025-compact-structures-collections-sets": "https://doi.org/10.4230/OASIcs.Grossi.6",
}
UA = {"User-Agent": "Snidewider-lit-fetch/1.0 (research; contact: repo owner)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


for name, url in ARXIV.items():
    dest = OUT / f"{name}.pdf"
    if dest.exists():
        continue
    try:
        dest.write_bytes(get(url))
        print(f"OK  {name} ({dest.stat().st_size // 1024} KB)", flush=True)
    except Exception as exc:
        print(f"ERR {name}: {exc}", flush=True)
    time.sleep(3)

for name, doi in DROPS.items():
    dest = OUT / f"{name}.pdf"
    if dest.exists():
        continue
    try:
        landing = get(doi).decode("utf-8", errors="ignore")
        m = re.search(r'href="([^"]+\.pdf)"', landing)
        if not m:
            print(f"ERR {name}: no pdf link on landing page", flush=True)
            continue
        pdf_url = m.group(1)
        if pdf_url.startswith("/"):
            pdf_url = "https://drops.dagstuhl.de" + pdf_url
        dest.write_bytes(get(pdf_url))
        print(f"OK  {name} ({dest.stat().st_size // 1024} KB)", flush=True)
    except Exception as exc:
        print(f"ERR {name}: {exc}", flush=True)
    time.sleep(3)

# paywalled at the publisher: hunt the preprints by title on arXiv
TITLE_QUERIES = {
    "liptak-masillo-puglisi2026-matching-statistics-survey":
        "Matching statistics a survey",
    "badkobeh2026-maximal-closed-substrings":
        "Finding maximal closed substrings",
    "alanko2025-finimizers":
        "Finimizers Variable-Length Bounded-Frequency Minimizers",
    "diseth2025-parallel-matching-statistics":
        "Massively Parallel Computation of Matching Statistics",
    "alanko2025-spectral-bwt-kmer-lookup":
        "Batched k-Mer Lookup on the Spectral Burrows-Wheeler Transform",
    "rizzo2025-elastic-founder-graphs":
        "Exploiting uniqueness seed-chain-extend alignment on elastic founder graphs",
}
for name, title in TITLE_QUERIES.items():
    dest = OUT / f"{name}.pdf"
    if dest.exists():
        continue
    try:
        q = urllib.parse.quote(f'ti:"{title}"')
        feed = get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=3").decode(
            "utf-8", errors="ignore")
        ids = re.findall(r"<id>http://arxiv.org/abs/([^v<]+)", feed)
        if not ids:
            print(f"ERR {name}: no arXiv preprint found", flush=True)
            continue
        dest.write_bytes(get(f"https://arxiv.org/pdf/{ids[0]}"))
        print(f"OK  {name} via arXiv:{ids[0]} ({dest.stat().st_size // 1024} KB)", flush=True)
    except Exception as exc:
        print(f"ERR {name}: {exc}", flush=True)
    time.sleep(3)

print("done")
