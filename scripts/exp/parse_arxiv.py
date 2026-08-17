"""Dump arXiv API result files as 'id | title | first 220 chars of abstract'."""
import re
import sys

for path in sys.argv[1:]:
    t = open(path, encoding="utf-8", errors="ignore").read()
    entries = re.findall(r"<entry>(.*?)</entry>", t, re.S)
    print(f"### {path}: {len(entries)} entries")
    for e in entries:
        aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        title = re.search(r"<title>(.*?)</title>", e, re.S)
        summ = re.search(r"<summary>(.*?)</summary>", e, re.S)
        pub = re.search(r"<published>([^<]+)</published>", e)
        title = re.sub(r"\s+", " ", title.group(1)).strip() if title else "?"
        summ = re.sub(r"\s+", " ", summ.group(1)).strip()[:220] if summ else ""
        print(f"{aid.group(1) if aid else '?'} | {pub.group(1)[:7] if pub else '?'} | "
              f"{title}\n    {summ}")
