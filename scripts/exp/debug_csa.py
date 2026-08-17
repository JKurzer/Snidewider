"""Debug the CSA SA output vs naive on a tiny string."""
from ai_text_detection import _csa_native

s = b"banana"
out = _csa_native.csa_stats(s)
print("native sa:", list(out["sa"]))
print("naive  sa:", sorted(range(len(s)), key=lambda i: s[i:]))
print("n reported:", out["n"])
print("bwt native:", bytes(out["bwt"]))
