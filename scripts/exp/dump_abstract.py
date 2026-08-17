"""Print title + abstract from arXiv abs pages."""
import re
import sys

for path in sys.argv[1:]:
    t = open(path, encoding="utf-8", errors="ignore").read()
    title = re.search(r'<meta name="citation_title" content="([^"]+)"', t)
    ab = re.search(r'<blockquote class="abstract mathjax">(.*?)</blockquote>', t, re.S)
    auth = re.findall(r'citation_author" content="([^"]+)"', t)[:4]
    if ab:
        txt = re.sub(r"<[^>]+>", " ", ab.group(1))
        txt = re.sub(r"\s+", " ", txt).strip()
    else:
        txt = "(no abstract found)"
    print(f"\n=== {title.group(1) if title else path} ===")
    print(f"    authors: {', '.join(auth)}")
    print(f"    {txt}")
