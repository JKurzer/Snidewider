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
#include <unordered_map>
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

// ---------------------------------------------------------------------------
// Similar substring search in the q-gram distance.
//
// Hanada, Kudo, Nakamura (TCS 530, 2014), "Array+Base-Search" (Fig. 5),
// built on Ukkonen's 1992 change-point machinery. The template flag
// kBaselineSkip selects Hanada's baseline trick (average-case O(|t|+|p|))
// or Ukkonen's original Array-Search (O(|t|k)) as the bench baseline.
//
// Problem (paper Sec. 2.2): for each start i, find j* in scope(i) minimizing
// d_q(t[i..j*], p), longest on ties; report it iff that distance <= k.
//   scope(i) = [i+|p|-1-k, i+|p|-1+k]  (only lengths |p| +/- k can be <= k)
//
// Change point (paper Def. 3 / Property 1): advancing i removes gram s = t(i);
// distances grow by +1 for j < c_i and shrink by -1 for j >= c_i, where c_i is
// the end position of the (#H_p(s)+1)-th occurrence of s in t[i..]. Occurrence
// lists + per-gram cursors make each c_i O(1); a hash map stands in for the
// paper's suffix tree (same O(1) role for window/pattern gram counts).
//
// Regime: k <= |p| - q keeps every in-scope substring >= q chars, the paper's
// working assumption (its delta formula starts at the first complete gram).
//
// Documented corner fix vs Fig. 5: when c_i falls outside the scope, the
// pseudocode never recomputes j*; if j* sat at the erased left edge it would
// dangle. We recompute the argmin in exactly that corner (oracle-verified).
namespace {

constexpr std::int64_t kInf = (1LL << 60);

struct SearchHit {
    std::int64_t start;
    std::int64_t end;  // exclusive
};

template <bool kBaselineSkip>
std::vector<SearchHit> qgram_search(const std::string& t, const std::string& p, int q,
                                    std::int64_t k) {
    check_q(q);
    const std::int64_t n = static_cast<std::int64_t>(t.size());
    const std::int64_t m = static_cast<std::int64_t>(p.size());
    if (m < q) throw std::invalid_argument("len(p) must be >= q");
    if (k < 0 || k > m - q) {
        throw std::invalid_argument("k must be in [0, len(p) - q]");
    }
    std::vector<SearchHit> hits;
    if (n < q) return hits;

    const std::vector<Code> codes_t = qgram_codes(t, q);
    const std::vector<Code> codes_p = qgram_codes(p, q);

    std::unordered_map<Code, std::int64_t> pat;
    for (const Code c : codes_p) ++pat[c];
    auto pcount = [&](Code g) -> std::int64_t {
        const auto it = pat.find(g);
        return it == pat.end() ? 0 : it->second;
    };

    std::unordered_map<Code, std::vector<std::int64_t>> occ;
    for (std::int64_t s = 0; s < static_cast<std::int64_t>(codes_t.size()); ++s) {
        occ[codes_t[s]].push_back(s);
    }
    std::unordered_map<Code, std::int64_t> seen;

    const std::int64_t width = 2 * k + 1;
    std::vector<std::int64_t> D(width, 0);  // offsets from baseline o, modular slots
    auto slot = [&](std::int64_t j) -> std::int64_t& { return D[j % width]; };
    // largest-j-on-ties argmin over [lo, hi]; requires lo <= hi
    auto argmin_over = [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t jj = lo, bv = slot(lo);
        for (std::int64_t j = lo + 1; j <= hi; ++j) {
            if (slot(j) <= bv) {
                bv = slot(j);
                jj = j;
            }
        }
        return jj;
    };

    std::unordered_map<Code, std::int64_t> wnd;  // gram counts in t[i..e]

    // ---- init at i = 0 (Fig. 5, lines 1-6) ----
    std::int64_t b = m - 1 - k;
    std::int64_t e = std::min(m - 1 + k, n - 1);
    if (b > e) return hits;
    std::int64_t cur_d = m - q + 1;  // d_q of the empty window vs pattern
    for (std::int64_t s = 0; s + q - 1 <= e; ++s) {
        const Code g = codes_t[s];
        const std::int64_t cnt = ++wnd[g];
        cur_d += (cnt <= pcount(g)) ? -1 : +1;
        const std::int64_t j = s + q - 1;
        if (j >= b) slot(j) = cur_d;
    }
    std::int64_t jstar = argmin_over(b, e);
    std::int64_t o = 0;
    if constexpr (kBaselineSkip) {
        o = slot(jstar);
        for (std::int64_t j = b; j <= e; ++j) slot(j) -= o;
    }
    if (o + slot(jstar) <= k) hits.push_back({0, jstar + 1});

    // ---- advance (Fig. 5, lines 7-23) ----
    const std::int64_t last_start = n - q;
    for (std::int64_t cur = 0; cur < last_start; ++cur) {
        // change point for the gram leaving at cur (Property 1)
        const Code s = codes_t[cur];
        const std::int64_t m_s = pcount(s);
        const auto& L = occ[s];
        const std::int64_t idx = seen[s]++ + m_s;
        const std::int64_t c = (idx < static_cast<std::int64_t>(L.size()))
                                   ? L[idx] + q - 1
                                   : kInf;

        const std::int64_t i = cur + 1;
        const std::int64_t e_prev = e;
        b = i + m - 1 - k;
        e = std::min(i + m - 1 + k, n - 1);
        if (b > e) break;  // scope ran off the end of t: no more hits possible

        const bool c_inside = (c >= b && c <= e);
        const bool full_update = !kBaselineSkip || c_inside;
        if (full_update) {  // full update path (Fig. 5, 12-13)
            for (std::int64_t j = b; j <= std::min(c - 1, e_prev); ++j) slot(j) += 1;
            for (std::int64_t j = std::max(c, b); j <= e_prev; ++j) slot(j) -= 1;
            if (b <= e_prev) jstar = argmin_over(b, e_prev);
        } else if (c < b) {  // baseline-only path (Fig. 5, 14-17)
            o -= 1;
        } else {
            o += 1;
        }
        if (jstar < b) {  // corner fix: j* erased at the left edge
            jstar = (b <= e_prev) ? argmin_over(b, e_prev) : b;
        }

        // sliding-window gram counts: out with cur, in with the new right end
        if (cur <= e_prev - q + 1) {
            if (--wnd[codes_t[cur]] == 0) wnd.erase(codes_t[cur]);
        }
        if (e > e_prev) {  // right edge (Fig. 5, 18, 22)
            // line 18 needs d(i)(e-1); on the full path with k=0 the update
            // loops skip e_prev (it is outside the single-slot scope), so the
            // slot still holds d(i-1)(e-1): apply the change-point delta now.
            // (Baseline path needs no fix: baseline shifts are offset-invariant.)
            if (full_update && e_prev < b) slot(e_prev) += (e_prev < c ? 1 : -1);
            const Code g = codes_t[e - q + 1];
            const std::int64_t cnt = ++wnd[g];
            slot(e) = slot(e - 1) + ((cnt <= pcount(g)) ? -1 : +1);
            if (slot(e) <= slot(jstar)) jstar = e;
        }
        if (o + slot(jstar) <= k) hits.push_back({i, jstar + 1});
    }
    return hits;
}

py::list search_impl(const py::bytes& t, const py::bytes& p, int q, std::int64_t k,
                     bool baseline_skip) {
    const std::string ts = t, ps = p;
    const std::vector<SearchHit> hits = baseline_skip
                                            ? qgram_search<true>(ts, ps, q, k)
                                            : qgram_search<false>(ts, ps, q, k);
    py::list out;
    for (const auto& h : hits) out.append(py::make_tuple(h.start, h.end));
    return out;
}

py::list search(const py::bytes& t, const py::bytes& p, int q, std::int64_t k) {
    return search_impl(t, p, q, k, true);
}

py::list search_ukkonen(const py::bytes& t, const py::bytes& p, int q, std::int64_t k) {
    return search_impl(t, p, q, k, false);
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
    m.def("search", &search, py::arg("t"), py::arg("p"), py::arg("q") = 5, py::arg("k"),
          "Hanada Array+Base-Search: all (start, end) with min d_q(t[start..end], p) <= k.");
    m.def("search_ukkonen", &search_ukkonen, py::arg("t"), py::arg("p"), py::arg("q") = 5,
          py::arg("k"),
          "Ukkonen Array-Search (O(|t|k)); same results, bench baseline.");
}
