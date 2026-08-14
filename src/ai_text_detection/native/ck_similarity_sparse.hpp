#pragma once
//linear time approximation of the lev dam distance.
//very very very fast.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

namespace ck::sparse {

inline constexpr std::size_t kArenaInlineBytes = 16 * 1024;

struct CKHash {
    std::array<std::uint32_t, 257> offset{};
    std::uint32_t* indices = nullptr;
    std::pmr::vector<unsigned char> present; // dense: distinct bytes, in first-seen order

    explicit CKHash(std::pmr::memory_resource* mr) : present(mr) {}

    std::span<const std::uint32_t> operator[](unsigned char c) const {
        return {indices + offset[c], indices + offset[static_cast<std::size_t>(c) + 1]};
    }
    bool contains(unsigned char c) const { return offset[c] != offset[static_cast<std::size_t>(c) + 1]; }
};

inline CKHash makeCKHash(const std::string& s, std::pmr::memory_resource* mr) {
    CKHash h(mr);
    if (s.empty()) return h;

    std::array<std::uint32_t, 256> counts{};
    for (unsigned char c : s) {
        if (counts[c]++ == 0) h.present.push_back(c); // first time this byte is seen
    }

    std::uint32_t running = 0;
    for (int c = 0; c < 256; ++c) {
        h.offset[static_cast<std::size_t>(c)] = running;
        running += counts[static_cast<std::size_t>(c)];
    }
    h.offset[256] = running;

    h.indices = static_cast<std::uint32_t*>(mr->allocate(s.size() * sizeof(std::uint32_t), alignof(std::uint32_t)));

    auto cursor = h.offset;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(s.size()); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        h.indices[cursor[c]++] = i;
    }
    return h;
}

struct CKMeasures {
    double Sa = 0.0;
    double Sb = 0.0;
    double D = 0.0;
    double S = 0.0;
    std::size_t n = 0;
};

// Precondition: X.length() == m <= Y.length() == n. Same algorithm/fixes as
// ck::corrected -- see that file for the derivation.
inline CKMeasures ckMeasuresSparse(const std::string& X, const std::string& Y, std::pmr::memory_resource* mr) {
    CKHash x = makeCKHash(X, mr);
    CKHash y = makeCKHash(Y, mr);

    double Sa = 0.0, Sb = 0.0, D = 0.0;
    std::pmr::vector<std::size_t> stack(mr);

    auto matchOne = [&](std::span<const std::uint32_t> xSpan, std::span<const std::uint32_t> ySpan) {
        bool swapped = false;
        auto shortSpan = xSpan, longSpan = ySpan;
        if (shortSpan.size() > longSpan.size()) {
            std::swap(shortSpan, longSpan);
            swapped = true;
        }

        std::size_t i = 0, j = 0;
        const std::size_t longSize = longSpan.size();
        stack.clear();

        while (i < shortSpan.size()) {
            const std::uint32_t xhead = shortSpan[i];
            ++i;

            while (j < longSize && longSpan[j] < xhead) {
                stack.push_back(longSpan[j]);
                ++j;
            }

            const bool sEmpty = stack.empty();
            const bool qyEmpty = (j >= longSize);

            if (!sEmpty && !qyEmpty) {
                const long long left = static_cast<long long>(xhead) - static_cast<long long>(stack.back());
                const long long right = static_cast<long long>(longSpan[j]) - static_cast<long long>(xhead);
                if (left < right) {
                    D += static_cast<double>(left);
                    stack.pop_back();
                } else {
                    D += static_cast<double>(right);
                    ++j;
                }
            } else if (!sEmpty) {
                D += static_cast<double>(xhead - stack.back());
                stack.pop_back();
            } else if (!qyEmpty) {
                D += static_cast<double>(longSpan[j] - xhead);
                ++j;
            }
        }

        const double leftover = static_cast<double>(stack.size()) + static_cast<double>(longSize - j);
        if (swapped) {
            Sa += leftover;
        } else {
            Sb += leftover;
        }
    };

    // x.keys ∩ y.keys, plus x.keys \ y.keys, both discovered by walking only
    // x's dense present-list (O(distinct chars in X), not O(256)).
    for (unsigned char c : x.present) {
        if (y.contains(c)) {
            matchOne(x[c], y[c]);
        } else {
            Sa += static_cast<double>(x[c].size());
        }
    }
    // y.keys \ x.keys: walking y's dense present-list, skipping anything
    // already handled by the intersection above.
    for (unsigned char c : y.present) {
        if (!x.contains(c)) Sb += static_cast<double>(y[c].size());
    }

    CKMeasures m;
    m.Sa = Sa;
    m.Sb = Sb;
    m.D = D;
    m.S = std::max(Sa, Sb);
    m.n = Y.size();
    return m;
}

inline double ckSimilarityFromMeasures(const CKMeasures& m) {
    const double n = static_cast<double>(m.n);
    if (n == 0.0) return 0.0;
    const double percentOrdered = (n * n - m.D) / (n * n);
    const double percentSimilarChars = (n - m.S) / n;
    return 1.0 - percentOrdered * percentSimilarChars;
}

inline double ckSimilarity(const std::string& A, const std::string& B) {
    const bool aFirst = (A.size() != B.size()) ? (A.size() < B.size()) : (A <= B);
    const std::string& X = aFirst ? A : B;
    const std::string& Y = aFirst ? B : A;

    alignas(alignof(std::max_align_t)) std::array<std::byte, kArenaInlineBytes> buffer;
    std::pmr::monotonic_buffer_resource arena(buffer.data(), buffer.size(), std::pmr::new_delete_resource());
    return ckSimilarityFromMeasures(ckMeasuresSparse(X, Y, &arena));
}

} // namespace ck::sparse
