// q-gram distance (Ukkonen 1992) — native implementation.
//
// Faithful port of the paper algorithm, not a re-invention:
//   profile(x) = count vector over all (|x|-q+1) substrings of length q
//   d_q(x,y)   = L1(profile(x), profile(y))          (Ukkonen 1992, Sec. 2)
//   d_bag(x,y) = max(P, N) where P,N are the + / - profile diffs at q=1
//                (Bartolini et al. 2002; the multiset/bag special case)
//
// Method: exact base-256 rolling codes (bijective for q<=8, so no collisions
// and no hash tables), then sort + merge-walk. No per-q-gram allocations.
// Byte-oriented: text encoding policy lives on the Python side.
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kMaxQ = 8;  // 256^8 == 2^64: codes stay exact in uint64

void check_q(int q) {
    if (q < 1 || q > kMaxQ) {
        throw std::invalid_argument("q must be in [1, 8]");
    }
}

using Code = std::uint64_t;
using Count = std::int64_t;
// A profile is sorted (code, count) runs — the compressed q-gram count vector.
using Profile = std::vector<std::pair<Code, Count>>;

std::vector<Code> qgram_codes(const std::string& s, int q) {
    const std::size_t n = s.size();
    if (n < static_cast<std::size_t>(q)) return {};
    const auto* b = reinterpret_cast<const unsigned char*>(s.data());

    std::vector<Code> codes(n - static_cast<std::size_t>(q) + 1);
    Code code = 0;
    for (int i = 0; i < q; ++i) code = code * 256u + b[i];
    codes[0] = code;

    Code drop = 1;  // 256^(q-1): weight of the outgoing byte
    for (int i = 1; i < q; ++i) drop *= 256u;
    for (std::size_t i = 1; i < codes.size(); ++i) {
        code = (code - b[i - 1] * drop) * 256u + b[i + q - 1];
        codes[i] = code;
    }
    return codes;
}

Profile make_profile(const std::string& s, int q) {
    auto codes = qgram_codes(s, q);
    std::sort(codes.begin(), codes.end());
    Profile prof;
    for (std::size_t i = 0; i < codes.size();) {
        std::size_t j = i + 1;
        while (j < codes.size() && codes[j] == codes[i]) ++j;
        prof.emplace_back(codes[i], static_cast<Count>(j - i));
        i = j;
    }
    return prof;
}

// Merge-walk two sorted profiles -> (P, N): sums of positive / negative
// count differences. Ukkonen d_q = P + N; bag distance = max(P, N).
std::pair<Count, Count> profile_diff(const Profile& x, const Profile& y) {
    Count pos = 0, neg = 0;
    std::size_t i = 0, j = 0;
    while (i < x.size() && j < y.size()) {
        if (x[i].first < y[j].first) {
            pos += x[i++].second;
        } else if (y[j].first < x[i].first) {
            neg += y[j++].second;
        } else {
            const Count d = x[i].second - y[j].second;
            if (d > 0) pos += d; else neg -= d;
            ++i;
            ++j;
        }
    }
    while (i < x.size()) pos += x[i++].second;
    while (j < y.size()) neg += y[j++].second;
    return {pos, neg};
}

Profile profile_from_py(const py::iterable& items) {
    Profile prof;
    for (py::handle item : items) {
        auto pair = item.cast<py::tuple>();
        prof.emplace_back(pair[0].cast<Code>(), pair[1].cast<Count>());
    }
    // Defensive: callers may hand us unsorted or unmerged runs.
    std::sort(prof.begin(), prof.end());
    Profile merged;
    for (const auto& [code, count] : prof) {
        if (!merged.empty() && merged.back().first == code) {
            merged.back().second += count;
        } else {
            merged.emplace_back(code, count);
        }
    }
    return merged;
}

py::list profile(const py::bytes& a, int q) {
    check_q(q);
    const std::string s = a;
    py::list out;
    for (const auto& [code, count] : make_profile(s, q)) {
        out.append(py::make_tuple(code, count));
    }
    return out;
}

std::pair<Count, Count> diff(const py::bytes& a, const py::bytes& b, int q) {
    check_q(q);
    return profile_diff(make_profile(std::string(a), q), make_profile(std::string(b), q));
}

std::int64_t distance(const py::bytes& a, const py::bytes& b, int q) {
    const auto [pos, neg] = diff(a, b, q);
    return pos + neg;
}

std::pair<std::int64_t, std::int64_t> diff_profiles(const py::iterable& x,
                                                    const py::iterable& y) {
    return profile_diff(profile_from_py(x), profile_from_py(y));
}

}  // namespace

PYBIND11_MODULE(_qgram_native, m) {
    m.doc() =
        "q-gram distance (Ukkonen 1992) + bag distance (Bartolini 2002).\n"
        "Raw byte semantics: no case folding, no whitespace stripping —\n"
        "normalization is a caller-side feature decision.\n"
        "distances are >= 0 (0 = identical profiles); larger = more different.";
    m.def("profile", &profile, py::arg("a"), py::arg("q") = 3,
          "Sorted (code, count) q-gram profile of a byte string.");
    m.def("diff", &diff, py::arg("a"), py::arg("b"), py::arg("q") = 3,
          "(P, N) profile count diffs between two byte strings.");
    m.def("distance", &distance, py::arg("a"), py::arg("b"), py::arg("q") = 3,
          "Ukkonen q-gram distance: P + N.");
    m.def("diff_profiles", &diff_profiles, py::arg("x"), py::arg("y"),
          "(P, N) diffs between two cached profiles (same q required).");
}
