"""One-shot: generate divsufsort.h from the cmake template (32-bit, static)."""
import re
from pathlib import Path

INC = Path("src/ai_text_detection/native/libdivsufsort/include")
tpl = (INC / "divsufsort.h.cmake").read_text(encoding="utf-8")

repl = {
    "@W64BIT@": "",
    "@INCFILE@": "#include <stdint.h>",
    "@DIVSUFSORT_EXPORT@": "",
    "@DIVSUFSORT_IMPORT@": "",
    "@SAUCHAR_TYPE@": "unsigned char",
    "@SAINT32_TYPE@": "int32_t",
    "@SAINDEX_TYPE@": "int32_t",
    "@SAINT_PRId@": '"d"',
    "@SAINDEX_PRId@": '"d"',
}
for k, v in repl.items():
    tpl = tpl.replace(k, v)
left = sorted(set(re.findall(r"@[A-Za-z0-9_]+@", tpl)))
print("leftover:", left)
(INC / "divsufsort.h").write_text(tpl, encoding="utf-8", newline="\n")
print("divsufsort.h generated")

# 64-bit variant header
tpl64 = (INC / "divsufsort.h.cmake").read_text(encoding="utf-8")
for k, v in [("@DIVSUFSORT_EXPORT@", ""), ("@DIVSUFSORT_IMPORT@", ""),
             ("@INCFILE@", "#include <stdint.h>"), ("@SAUCHAR_TYPE@", "unsigned char"),
             ("@SAINT32_TYPE@", "int32_t"), ("@SAINDEX_TYPE@", "int64_t"),
             ("@SAINT_PRId@", '"d"'), ("@SAINDEX_PRId@", '"lld"'), ("@W64BIT@", "64")]:
    tpl64 = tpl64.replace(k, v)
(INC / "divsufsort64.h").write_text(tpl64, encoding="utf-8", newline="\n")
print("divsufsort64.h generated")

# forwarding TUs for the 64-bit variant (mirrors cmake's per-file macro build)
LIB = Path("src/ai_text_detection/native/libdivsufsort/lib")
for stem in ("divsufsort", "sssort", "trsort", "utils"):
    (LIB / f"dss64_{stem}.c").write_text(
        f'#define BUILD_DIVSUFSORT64\n#include "{stem}.c"\n',
        encoding="utf-8", newline="\n")
print("4 forwarding units written")
