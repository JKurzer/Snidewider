"""Build the native extensions in-place (dev convenience).

Compiles native/*.cpp with the vendored headers and drops the built
modules next to the package so imports pick up the fast paths.

Usage: .venv\\Scripts\\python scripts/build_native.py
"""

import shutil
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "src" / "ai_text_detection" / "native"
PKG = ROOT / "src" / "ai_text_detection"
L0 = NATIVE / "l0learn"
SDSL = NATIVE / "sdsl"
LDSS = NATIVE / "libdivsufsort"

EXTENSIONS = [
    Pybind11Extension(
        "_ck2_native",
        sources=[str(NATIVE / "ck2_module.cpp")],
        include_dirs=[str(NATIVE)],
        cxx_std=20,  # ck headers use std::span
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
    ),
    Pybind11Extension(
        "_qgram_native",
        sources=[str(NATIVE / "qgram_module.cpp")],
        cxx_std=17,
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
    ),
    Pybind11Extension(
        "_l0learn_native",
        sources=[str(NATIVE / "l0learn_module.cpp")]
        + [str(L0 / f) for f in (
            "BetaVector.cpp", "CDL012LogisticSwaps.cpp", "CDL012SquaredHingeSwaps.cpp",
            "CDL012Swaps.cpp", "Grid.cpp", "Grid1D.cpp", "Grid2D.cpp",
            "Normalize.cpp", "utils.cpp",
        )],
        include_dirs=[str(L0), str(L0 / "include"), str(L0 / "arma_include")],
        cxx_std=17,
        extra_compile_args=["/O2", "/DARMA_DONT_USE_WRAPPER", "/DARMA_DONT_USE_BLAS",
                            "/DARMA_DONT_USE_LAPACK", "/DARMA_DONT_USE_OPENMP"]
        if sys.platform == "win32"
        else ["-O3", "-DARMA_DONT_USE_WRAPPER", "-DARMA_DONT_USE_BLAS",
              "-DARMA_DONT_USE_LAPACK", "-DARMA_DONT_USE_OPENMP"],
    ),
    Pybind11Extension(
        "_csa_native",
        sources=[str(NATIVE / "csa_module.cpp")]
        + [str(p) for p in (SDSL / "lib").glob("*.cpp")]
        + [str(p) for p in (LDSS / "lib").glob("*.c")],
        include_dirs=[str(SDSL / "include"), str(LDSS / "include")],
        cxx_std=17,
        # MSVC_COMPILER: sdsl's own guard macro - skips POSIX headers on
        # Windows builds. /permissive- makes alternative tokens (or/and)
        # real keywords for sdsl's headers.
        extra_compile_args=["/O2", "/permissive-", "/DMSVC_COMPILER", "/DHAVE_CONFIG_H"]
        if sys.platform == "win32" else ["-O3"],
    ),
]


def main() -> int:
    setup(
        name="native-build",
        ext_modules=EXTENSIONS,
        script_args=["build_ext", "--inplace"],
        cmdclass={"build_ext": build_ext},
    )
    ok = True
    for ext in EXTENSIONS:
        built = list(ROOT.glob(f"{ext.name}*.pyd")) + list((ROOT / "src").glob(f"{ext.name}*.pyd")) + list(
            ROOT.glob(f"{ext.name}*.so")
        )
        if not built:
            print(f"ERROR: no built artifact found for {ext.name}", file=sys.stderr)
            ok = False
            continue
        for artifact in built:
            target = PKG / artifact.name
            shutil.copy2(artifact, target)
            artifact.unlink()
            print(f"OK: {target}")
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
