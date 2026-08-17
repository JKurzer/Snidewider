"""Print a DBLP author XML's publications since a year."""
import re
import sys

path, since = sys.argv[1], int(sys.argv[2])
t = open(path, encoding="utf-8", errors="ignore").read()
records = re.findall(r"<r>(.*?)</r>", t, re.S)
out = []
for r in records:
    y = re.search(r"<year>(\d+)</year>", r)
    ti = re.search(r"<title>(.*?)</title>", r, re.S)
    aus = re.findall(r"<author[^>]*>(.*?)</author>", r)
    if y and ti and int(y.group(1)) >= since:
        out.append((int(y.group(1)), re.sub(r"\s+", " ", ti.group(1)).strip(),
                    ", ".join(aus[:4])))
out.sort(reverse=True)
for y, ti, aus in out:
    print(f"{y} | {ti[:110]} | {aus}")
print(f"-- {len(out)} pubs >= {since}")
