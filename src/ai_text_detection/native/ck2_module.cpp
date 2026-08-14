// Python bindings for CK2 (ck::sparse8 / ck::sparse) — the CKS linear-time
// Levenshtein approximation. Thin shim only; all logic lives in the vendored
// headers. Byte-oriented: encode text to UTF-8 on the Python side.
#include <pybind11/pybind11.h>

#include <string>

#include "ck_similarity_sparse8.hpp"

namespace py = pybind11;

namespace {

py::dict measures(const py::bytes& a, const py::bytes& b) {
    std::string A = a, B = b;
    const bool aFirst = (A.size() != B.size()) ? (A.size() < B.size()) : (A <= B);
    const std::string& X = aFirst ? A : B;
    const std::string& Y = aFirst ? B : A;

    alignas(alignof(std::max_align_t)) std::array<std::byte, ck::sparse::kArenaInlineBytes> buffer;
    // Measures path uses the general uint32 arena implementation; sparse8 is
    // the fast path for similarity() only.
    std::pmr::monotonic_buffer_resource arena(buffer.data(), buffer.size(),
                                              std::pmr::new_delete_resource());
    const auto m = ck::sparse::ckMeasuresSparse(X, Y, &arena);
    py::dict d;
    d["Sa"] = m.Sa;
    d["Sb"] = m.Sb;
    d["D"] = m.D;
    d["S"] = m.S;
    d["n"] = m.n;
    d["score"] = ck::sparse::ckSimilarityFromMeasures(m);
    return d;
}

double similarity(const py::bytes& a, const py::bytes& b) {
    const std::string A = a, B = b;
    return ck::sparse8::ckSimilarity(A, B);  // auto-falls-back to sparse >=255 B
}

}  // namespace

PYBIND11_MODULE(_ck2_native, m) {
    m.doc() = "CK2/CKS native bindings: linear-time Levenshtein approximation. "
              "Score semantics: 0.0 = identical, 1.0 = maximally different "
              "(despite the historical 'similarity' name).";
    m.def("similarity", &similarity, py::arg("a"), py::arg("b"),
          "CK2 score between two byte strings in [0, 1].");
    m.def("measures", &measures, py::arg("a"), py::arg("b"),
          "Intermediate CK measures (Sa, Sb, D, S, n) plus score, for verification.");
}
