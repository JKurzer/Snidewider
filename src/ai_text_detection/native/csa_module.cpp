// pybind11 wrapper for sdsl-lite compressed suffix arrays.
// The point: the BEHAVIOR (compressed size) of a document's CSA is a
// Kolmogorov-flavored statistic (docs/condensates.md, TODO #0). Also returns
// the plain SA for downstream features + oracle verification.
#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <sdsl/csa_sada.hpp>
#include <sdsl/csa_wt.hpp>
#include <sdsl/ram_fs.hpp>
#include <sdsl/suffix_arrays.hpp>
#include <string>

namespace py = pybind11;

namespace {

py::dict csa_stats(const std::string& s) {
    py::dict out;

    // Shared cache_config: the first construct registers SA/BWT/ISA in the
    // file_map; the second REUSES them (halves the work per doc). ram_fs is
    // purged per call below.
    const std::string id = sdsl::util::to_string(sdsl::util::pid()) + "_" +
                           sdsl::util::to_string(sdsl::util::id());
    sdsl::cache_config config(false, "@", id);  // delete_data=false: share intermediates
    const std::string tmp = sdsl::ram_file_name(id);
    sdsl::store_to_file(s, tmp);

    sdsl::csa_wt<sdsl::wt_huff<>> wt;
    sdsl::construct(wt, tmp, config, 1);
    out["csa_wt_bytes"] = static_cast<double>(sdsl::size_in_bytes(wt));

    sdsl::csa_sada<> sada;
    sdsl::construct(sada, tmp, config, 1);
    out["csa_sada_bytes"] = static_cast<double>(sdsl::size_in_bytes(sada));

    // purge every ram_fs entry we created (registry is process-global)
    sdsl::ram_fs::remove(tmp);
    for (const auto& kv : config.file_map) sdsl::ram_fs::remove(kv.second);

    // NB: sdsl CSAs cover s + implicit sentinel, so size() == len(s) + 1 and
    // sa[0] == len(s) (the sentinel suffix). Callers get the full n+1 arrays.
    const std::size_t n = wt.size();
    out["n"] = n;
    py::array_t<int64_t> sa(static_cast<std::ptrdiff_t>(n));
    auto sam = sa.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) sam(i) = static_cast<int64_t>(wt[i]);
    out["sa"] = sa;

    // BWT over the sentinel-terminated string (p == 0 contributes the
    // sentinel's predecessor = last char; p == n-1+... sentinel row gets 0)
    py::array_t<uint8_t> bwt(static_cast<std::ptrdiff_t>(n));
    auto bm = bwt.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t p = static_cast<std::size_t>(wt[i]);
        bm(i) = static_cast<uint8_t>(p == 0 ? 0 : s[p - 1]);
    }
    out["bwt"] = bwt;
    return out;
}

}  // namespace

namespace {

py::dict bench_construct(const std::string& s, int iters) {
    py::dict out;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sdsl::csa_wt<sdsl::wt_huff<>> wt;
        sdsl::construct_im(wt, s, 1);
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sdsl::csa_sada<> sada;
        sdsl::construct_im(sada, s, 1);
    }
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        py::dict d = csa_stats(s);
    }
    auto t3 = std::chrono::steady_clock::now();
    out["wt_ms"] = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    out["sada_ms"] = std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
    out["shared_ms"] = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;
    return out;
}

}  // namespace

PYBIND11_MODULE(_csa_native, m) {
    m.doc() = "sdsl-lite CSA measures (csa_wt<wt_huff> + csa_sada sizes, SA, BWT).";
    m.def("csa_stats", &csa_stats, py::arg("s"),
          "Build CSAs over s; return sizes in bytes, the suffix array, and the BWT.");
    m.def("ramfs_size", []() { return sdsl::ram_fs::size(); },
          "Diagnostic: number of files in sdsl's ram_fs registry (leak watch).");
    m.def("ramfs_keys", []() { return sdsl::ram_fs::keys(); },
          "Diagnostic: current ram_fs keys.");
    m.def("bench_construct", &bench_construct, py::arg("s"), py::arg("iters") = 50,
          "Micro-bench: ms per construct_im for csa_wt and csa_sada.");
}
