"""Peek at the RAID full-test file size before committing to the run."""
from urllib.request import urlopen

for name in ("test_none.csv", "test.csv"):
    r = urlopen(f"https://dataset.raid-bench.xyz/{name}")
    size = int(r.headers["Content-Length"])
    r.close()
    print(f"{name}: {size/1e6:.1f} MB")
