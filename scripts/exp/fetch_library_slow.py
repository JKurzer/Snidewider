"""Patient serial fetcher for the rate-limited stragglers (arXiv API, long backoff)."""
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("papers")
UA = {"User-Agent": "Snidewider-lit-fetch/1.0 (research)"}

TITLES = {
    "liptak-masillo-puglisi2026-matching-statistics-survey":
        "Matching statistics a survey Liptak",
    "badkobeh2026-maximal-closed-substrings":
        "Finding maximal closed substrings",
    "alanko2025-finimizers":
        "Finimizers Variable-Length Bounded-Frequency Minimizers",
    "diseth2025-parallel-matching-statistics":
        "Massively Parallel Computation of Matching Statistics",
    "alanko2025-spectral-bwt-kmer-lookup":
        "Batched k-Mer Lookup Spectral Burrows-Wheeler",
    "rizzo2025-elastic-founder-graphs":
        "seed-chain-extend alignment elastic founder graphs",
}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


for name, title in TITLES.items():
    dest = OUT / f"{name}.pdf"
    if dest.exists():
        print(f"SKIP {name} (exists)", flush=True)
        continue
    for attempt in range(4):
        try:
            q = urllib.parse.quote(f'ti:"{title}"')
            feed = get("http://export.arxiv.org/api/query"
                       f"?search_query={q}&max_results=3").decode("utf-8", errors="ignore")
            ids = re.findall(r"<id>http://arxiv.org/abs/([^v<]+)", feed)
            if not ids:
                print(f"ERR {name}: no preprint", flush=True)
                break
            dest.write_bytes(get(f"https://arxiv.org/pdf/{ids[0]}"))
            print(f"OK  {name} via arXiv:{ids[0]} ({dest.stat().st_size // 1024} KB)", flush=True)
            break
        except Exception as exc:
            wait = 45 * (attempt + 1)
            print(f"  retry {name} in {wait}s ({exc})", flush=True)
            time.sleep(wait)
    time.sleep(20)

print("done")
