"""Leak test: 20K csa_stats calls on one small doc. Survival is the verdict."""
from ai_text_detection import _csa_native

b = b"Whoa, Nelly! is the debut studio album by Canadian singer-songwriter." * 3
LOG = r"scripts\exp\_csa_leak.log"

with open(LOG, "w", encoding="utf-8") as fh:
    for i in range(20_000):
        _ = _csa_native.csa_stats(b)
        if i % 1000 == 0:
            keys = _csa_native.ramfs_keys()
            fh.write(f"{i} ramfs={_csa_native.ramfs_size()} sample_keys={keys[:4]}\n")
            fh.flush()
    fh.write("SURVIVED 20000 ramfs=%d\n" % _csa_native.ramfs_size())
print("done")
