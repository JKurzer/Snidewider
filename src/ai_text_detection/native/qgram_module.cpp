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

#include <pybind11/numpy.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
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

// ---------------------------------------------------------------------------
// Bank registry: exemplar-style batch scoring without per-call marshalling.
// Profiles live here once; each doc pays ONE conversion of its own profile.
struct BankSet {
    std::vector<Profile> profiles;
    std::vector<Count> totals;
};
BankSet g_bank_ai, g_bank_hu;
std::string g_bank_key;

BankSet bank_from_py(const py::iterable& profiles, const py::iterable& totals) {
    BankSet b;
    for (const auto& p : profiles) b.profiles.push_back(profile_from_py(p.cast<py::iterable>()));
    for (const auto& t : totals) b.totals.push_back(t.cast<Count>());
    return b;
}

void load_banks(const py::iterable& ai_profiles, const py::iterable& ai_totals,
                const py::iterable& hu_profiles, const py::iterable& hu_totals,
                const std::string& key) {
    g_bank_ai = bank_from_py(ai_profiles, ai_totals);
    g_bank_hu = bank_from_py(hu_profiles, hu_totals);
    g_bank_key = key;
}

std::string banks_key() { return g_bank_key; }

py::dict bank_distances(const py::iterable& doc_profile, std::int64_t doc_total,
                        std::int64_t ai_skip, std::int64_t hu_skip) {
    const Profile doc = profile_from_py(doc_profile);
    py::dict out;
    for (int side = 0; side < 2; ++side) {
        const BankSet& bank = side == 0 ? g_bank_ai : g_bank_hu;
        const std::int64_t skip = side == 0 ? ai_skip : hu_skip;
        const std::size_t m = bank.profiles.size();
        py::array_t<double> raw(std::vector<std::ptrdiff_t>{(std::ptrdiff_t)m});
        py::array_t<double> norm(std::vector<std::ptrdiff_t>{(std::ptrdiff_t)m});
        auto rm = raw.mutable_unchecked<1>();
        auto nm = norm.mutable_unchecked<1>();
        for (std::size_t i = 0; i < m; ++i) {
            if ((std::int64_t)i == skip) { rm(i) = std::nan(""); nm(i) = std::nan(""); continue; }
            const auto [pos, neg] = profile_diff(doc, bank.profiles[i]);
            const double d = static_cast<double>(pos + neg);
            const double denom = static_cast<double>(doc_total + bank.totals[i]);
            rm(i) = d;
            nm(i) = denom > 0 ? d / denom : 0.0;
        }
        out[side == 0 ? "ai_raw" : "hu_raw"] = raw;
        out[side == 0 ? "ai_norm" : "hu_norm"] = norm;
    }
    return out;
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

// Profiling counters (PERF-RULES #2: instrument first). When non-null,
// qgram_search fills these so we can see WHERE the time goes per path.
struct SearchStats {
    std::int64_t steps = 0;
    std::int64_t full_updates = 0;
    std::int64_t baseline_steps = 0;
    std::int64_t evictions = 0;      // deque front pops (left-edge leaves scope)
    std::int64_t back_pops = 0;      // deque insertions evicting worse candidates
    std::int64_t update_iters = 0;   // total slot +=/-= work
    std::int64_t argmin_iters = 0;   // total argmin/deque-rebuild scan work
    std::int64_t edge_computes = 0;
    std::int64_t init_iters = 0;
    double init_ns = 0.0;
    double loop_ns = 0.0;
};

template <bool kBaselineSkip>
std::vector<SearchHit> qgram_search(const std::string& t, const std::string& p, int q,
                                    std::int64_t k, SearchStats* stats = nullptr) {
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

    // Hanada path: sliding-window min as a monotone deque of (j, offset),
    // front = min offset with largest-j-on-ties. Frozen relative order on the
    // baseline path (uniform o shifts) makes eviction/insertion amortized O(1);
    // full updates rebuild it O(k), charged to alpha. This is the paper's own
    // List+Base candidate-list idea (Sec. 4.3, DEL-FIRST eviction) applied to
    // the Array variant -- and it replaces the argmin corner fix entirely.
    std::deque<std::pair<std::int64_t, std::int64_t>> cand;
    auto deque_push = [&](std::int64_t j) {
        while (!cand.empty() && cand.back().second >= slot(j)) {
            cand.pop_back();
            if (stats) ++stats->back_pops;
        }
        cand.emplace_back(j, slot(j));
    };
    auto deque_rebuild = [&](std::int64_t lo, std::int64_t hi) {
        cand.clear();
        for (std::int64_t j = lo; j <= hi; ++j) {
            deque_push(j);
            if (stats) ++stats->argmin_iters;
        }
    };

    std::unordered_map<Code, std::int64_t> wnd;  // gram counts in t[i..e]

    const auto t_init0 = std::chrono::steady_clock::now();

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
        if (stats) ++stats->init_iters;
    }
    std::int64_t jstar = argmin_over(b, e);
    if (stats) stats->argmin_iters += e - b;
    std::int64_t o = 0;
    if constexpr (kBaselineSkip) {
        o = slot(jstar);
        for (std::int64_t j = b; j <= e; ++j) slot(j) -= o;
        deque_rebuild(b, e);
        jstar = cand.front().first;
    }
    if (o + slot(jstar) <= k) hits.push_back({0, jstar + 1});
    if (stats) {
        stats->init_ns = std::chrono::duration<double, std::nano>(
                             std::chrono::steady_clock::now() - t_init0)
                             .count();
    }

    // ---- advance (Fig. 5, lines 7-23) ----
    const auto t_loop0 = std::chrono::steady_clock::now();
    const std::int64_t last_start = n - q;
    for (std::int64_t cur = 0; cur < last_start; ++cur) {
        if (stats) ++stats->steps;
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
            const std::int64_t up_hi = std::min(c - 1, e_prev);
            for (std::int64_t j = b; j <= up_hi; ++j) slot(j) += 1;
            const std::int64_t dn_lo = std::max(c, b);
            for (std::int64_t j = dn_lo; j <= e_prev; ++j) slot(j) -= 1;
            if (stats) {
                ++stats->full_updates;
                stats->update_iters +=
                    (up_hi >= b ? up_hi - b + 1 : 0) + (e_prev >= dn_lo ? e_prev - dn_lo + 1 : 0);
                if (b <= e_prev) stats->argmin_iters += e_prev - b;
            }
            if constexpr (kBaselineSkip) {
                deque_rebuild(b, e_prev);  // relative order changed: O(k) rebuild
            } else if (b <= e_prev) {
                jstar = argmin_over(b, e_prev);
            }
        } else if (c < b) {  // baseline-only path (Fig. 5, 14-17)
            o -= 1;
            if (stats) ++stats->baseline_steps;
        } else {
            o += 1;
            if (stats) ++stats->baseline_steps;
        }
        if constexpr (kBaselineSkip) {
            if (!full_update) {  // structural eviction (DEL-FIRST analogue)
                while (!cand.empty() && cand.front().first < b) {
                    cand.pop_front();
                    if (stats) ++stats->evictions;
                }
            }
        } else if (jstar < b) {  // corner fix: j* erased at the left edge
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
            if constexpr (kBaselineSkip) {
                deque_push(e);
                jstar = cand.front().first;
            } else if (slot(e) <= slot(jstar)) {
                jstar = e;
            }
            if (stats) ++stats->edge_computes;
        } else if constexpr (kBaselineSkip) {
            jstar = cand.front().first;
        }
        if (o + slot(jstar) <= k) hits.push_back({i, jstar + 1});
    }
    if (stats) {
        stats->loop_ns = std::chrono::duration<double, std::nano>(
                             std::chrono::steady_clock::now() - t_loop0)
                             .count();
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

py::dict search_debug(const py::bytes& t, const py::bytes& p, int q, std::int64_t k) {
    const std::string ts = t, ps = p;
    SearchStats stats;
    const auto hits = qgram_search<true>(ts, ps, q, k, &stats);
    py::dict d;
    d["hits"] = hits.size();
    d["steps"] = stats.steps;
    d["full_updates"] = stats.full_updates;
    d["baseline_steps"] = stats.baseline_steps;
    d["evictions"] = stats.evictions;
    d["back_pops"] = stats.back_pops;
    d["update_iters"] = stats.update_iters;
    d["argmin_iters"] = stats.argmin_iters;
    d["edge_computes"] = stats.edge_computes;
    d["init_iters"] = stats.init_iters;
    d["init_ms"] = stats.init_ns / 1e6;
    d["loop_ms"] = stats.loop_ns / 1e6;
    return d;
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
    m.def("search_debug", &search_debug, py::arg("t"), py::arg("p"), py::arg("q") = 5,
          py::arg("k"), "Hanada search with profiling counters (PERF-RULES instrumentation).");
    m.def("load_banks", &load_banks, py::arg("ai_profiles"), py::arg("ai_totals"),
          py::arg("hu_profiles"), py::arg("hu_totals"), py::arg("key"),
          "Load exemplar banks into the process-resident registry (once per process).");
    m.def("banks_key", &banks_key, "Registry key of the currently loaded banks.");
    m.def("bank_distances", &bank_distances, py::arg("doc_profile"), py::arg("doc_total"),
          py::arg("ai_skip") = -1, py::arg("hu_skip") = -1,
          "Raw + normalized distances from one doc profile to every bank exemplar.");
}
