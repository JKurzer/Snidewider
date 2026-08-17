"""Dump year | venue | title | ee/links from a DBLP publ-API XML, since a year."""
import re
import sys

path, since = sys.argv[1], int(sys.argv[2])
t = open(path, encoding="utf-8", errors="ignore").read()
hits = re.findall(r"<hit[^>]*>(.*?)</hit>", t, re.S)
rows = []
for h in hits:
    m = re.search(r"<info>(.*?)</info>", h, re.S)
    if not m:
        continue
    info = m.group(1)
    y = re.search(r"<year>(\d+)</year>", info)
    ti = re.search(r"<title>(.*?)</title>", info, re.S)
    ve = re.search(r"<venue>(.*?)</venue>", info, re.S)
    ees = re.findall(r"<ee[^>]*>(.*?)</ee>", info)
    if y and ti and int(y.group(1)) >= since:
        title = re.sub(r"\s+", " ", ti.group(1)).strip().rstrip(".")
        rows.append((int(y.group(1)), ve.group(1) if ve else "?",
                     title, "; ".join(ees)))
rows.sort(reverse=True)
for y, venue, title, ee in rows:
    print(f"{y} | {venue} | {title}\n    {ee}")
print(f"-- {len(rows)} records")
