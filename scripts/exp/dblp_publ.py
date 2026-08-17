"""List year | venue | title | authors from a DBLP publ-API XML, since a year."""
import re
import sys

path, since = sys.argv[1], int(sys.argv[2])
t = open(path, encoding="utf-8", errors="ignore").read()
hits = re.findall(r"<hit[^>]*>(.*?)</hit>", t, re.S)
rows = []
for h in hits:
    info = re.search(r"<info>(.*?)</info>", h, re.S)
    if not info:
        continue
    info = info.group(1)
    y = re.search(r"<year>(\d+)</year>", info)
    ti = re.search(r"<title>(.*?)</title>", info, re.S)
    ve = re.search(r"<venue>(.*?)</venue>", info, re.S)
    aus = re.findall(r"<author[^>]*>(.*?)</author>", info)
    if y and ti and int(y.group(1)) >= since:
        title = re.sub(r"\s+", " ", ti.group(1)).strip().rstrip(".")
        venue = ve.group(1) if ve else "?"
        rows.append((int(y.group(1)), venue, title, ", ".join(aus)))
rows.sort(reverse=True)
for y, venue, title, aus in rows:
    print(f"{y} | {venue:<28} | {title[:95]} | {aus[:90]}")
print(f"-- {len(rows)} pubs >= {since} (of {len(hits)} returned)")
