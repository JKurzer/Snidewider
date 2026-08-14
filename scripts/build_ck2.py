"""Build the _ck2_native extension in-place (dev convenience).

Proper packaging can come later; this compiles native/ck2_module.cpp with the
vendored headers and drops _ck2_native.*.pyd next to the package so
`import ai_text_detection.ck2` picks up the fast path.

Usage: .venv\\Scripts\\python scripts/build_ck2.py
"""

import shutil
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "src" / "ai_text_detection" / "native"
PKG = ROOT / "src" / "ai_text_detection"

ext = Pybind11Extension(
    "_ck2_native",
    sources=[str(NATIVE / "ck2_module.cpp")],
    include_dirs=[str(NATIVE)],
    cxx_std=20,  # headers use std::span
    extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
)


def main() -> int:
    setup(
        name="ck2-native-build",
        ext_modules=[ext],
        script_args=["build_ext", "--inplace"],
        cmdclass={"build_ext": build_ext},
    )
    built = list(ROOT.glob("_ck2_native*.pyd")) + list((ROOT / "src").glob("_ck2_native*.pyd")) + list(
        ROOT.glob("_ck2_native*.so")
    )
    if not built:
        print("ERROR: no built artifact found", file=sys.stderr)
        return 1
    for artifact in built:
        target = PKG / artifact.name
        shutil.copy2(artifact, target)
        artifact.unlink()
        print(f"OK: {target}")
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
