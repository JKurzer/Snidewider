"""List author-name -> pid hits from DBLP author-API XML files."""
import re
import sys

for path in sys.argv[1:]:
    t = open(path, encoding="utf-8", errors="ignore").read()
    hits = re.findall(r'<info><author>(.*?)</author>(.*?)</info>', t, re.S)
    for name, rest in hits:
        url = re.search(r"<url>(.*?)</url>", rest)
        print(f"  {name} | {url.group(1) if url else '?'}")
    print(f"-- {path.split(chr(92))[-1]}: {len(hits)} hits")
