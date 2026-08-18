"""Fetch and display the RAID leaderboard."""
import re
import urllib.request

req = urllib.request.Request("https://raid-bench.xyz/",
                             headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=30).read()
text = html.decode("utf-8", errors="ignore")
print(len(text), "bytes of html")
links = sorted(set(re.findall(r'href="([^"]+)"', text)))
print("links:", links[:30])
plain = re.sub(r"<[^>]+>", " ", text)
plain = re.sub(r"\s+", " ", plain)
print(plain[:4000])
