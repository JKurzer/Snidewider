#pragma once
// Only covers len < 255 for BOTH strings -- ckSimilarity below falls back
// to ck::sparse (uint32_t, unrestricted -- confirmed to be the right
// general-purpose choice past this point) whenever either string doesn't
// fit that bound. This uses a classic Briggs & Torczon sparse set, same as SuperSparseSets,
//  but flat/index-based here.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <intrin.h> // __stosb
#include <malloc.h> // alloca
#include <span>
#include <string>

#include "ck_similarity_sparse.hpp" // fallback for len >= 255

namespace ck::sparse8 {

inline constexpr std::size_t kMaxLen = 255; // strict: length must be < kMaxLen

struct CKHash {
    // `sparse[c]` is a CANDIDATE index into `present`/spanStart/spanCount --
    // deliberately never initialized. Only ever trusted after validating
    // idx < presentCount && present[idx] == c (see contains()/operator[]
    // below), so an unwritten entry (garbage) just fails that check instead
    // of needing to be zeroed first. This is the whole point of the
    // technique: O(1) lookup with zero construction cost.
    std::array<std::uint8_t, 256> sparse;
    // Parallel to `present`: spanStart[i]/spanCount[i] describe present[i]'s
    // slice of `indices`. Sized to the worst case (256, if every byte value
    // occurs at least once) but only [0, presentCount) is ever written or
    // read -- same "only touch what's actually used" reasoning as
    // `present` itself, no zero-init needed.
    std::array<std::uint8_t, 256> spanStart;
    std::array<std::uint8_t, 256> spanCount;
    std::uint8_t* indices = nullptr;       // caller-owned (alloca'd), not freed here
    const std::uint8_t* present = nullptr; // caller-owned, fixed 256-byte buffer
    std::uint8_t presentCount = 0;

    std::span<const std::uint8_t> operator[](unsigned char c) const {
        const std::uint8_t idx = sparse[c];
        if (idx < presentCount && present[idx] == c) {
            return {indices + spanStart[idx], indices + spanStart[idx] + spanCount[idx]};
        }
        return {};
    }
    bool contains(unsigned char c) const {
        const std::uint8_t idx = sparse[c];
        return idx < presentCount && present[idx] == c;
    }

    std::span<const std::uint8_t> spanAt(std::size_t idx) const {
        return {indices + spanStart[idx], indices + spanStart[idx] + spanCount[idx]};
    }
};

// Precondition: s.size() < kMaxLen (255). indicesBuf must point to at least
// s.size() bytes (typically alloca'd by the caller, or nullptr if s is
// empty -- never dereferenced in that case since every span ends up
// empty); presentBuf must point to at least 256 bytes (a plain fixed-size
// local array is enough -- see ckMeasuresSparse8). Both must outlive the
// returned CKHash.
inline CKHash fillCKHash(const std::string& s, std::uint8_t* indicesBuf, std::uint8_t* presentBuf) {
    CKHash h;
    h.indices = indicesBuf;
    h.present = presentBuf;


    std::array<std::uint8_t, 256> counts;
    __stosb(counts.data(), 0, counts.size());
    for (unsigned char c : s) {
        if (counts[c]++ == 0) {
            h.sparse[c] = h.presentCount; // record c's future dense-index, no init needed first
            presentBuf[h.presentCount++] = c;
        }
    }

    std::uint8_t running = 0;
    for (std::uint8_t i = 0; i < h.presentCount; ++i) {
        const unsigned char c = presentBuf[i];
        h.spanStart[i] = running;
        h.spanCount[i] = counts[c];
        running += counts[c];
        counts[c] = h.spanStart[i];
    }

    // size_t loop counter, not uint8_t: s.size() can legitimately be 254,
    // and a uint8_t counter comparing against a wrapped cast of 255 would be
    // an easy off-by-one/wraparound bug. Only the STORED index needs to be
    // narrow; the loop control doesn't.
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        h.indices[counts[c]++] = static_cast<std::uint8_t>(i);
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

// Precondition: X.length() == m <= Y.length() == n < kMaxLen. Same
// algorithm/fixes as ck::corrected -- see that file for the derivation.
inline CKMeasures ckMeasuresSparse8(const std::string& X, const std::string& Y) {
    // alloca must be called directly in the frame that needs the memory to
    // stay alive -- it can't be wrapped in a helper function. One call per
    // side; 0-length strings skip it (fillCKHash never dereferences
    // indices when s.empty(), so nullptr is fine there).
    std::uint8_t* xIndices = X.empty() ? nullptr : static_cast<std::uint8_t*>(alloca(X.size()));
    std::uint8_t* yIndices = Y.empty() ? nullptr : static_cast<std::uint8_t*>(alloca(Y.size()));

    // Fixed, compile-time-sized: every one of these is bounded by kMaxLen
    // (255) regardless of the actual string length, so there's nothing to
    // alloca here -- a plain local array already has the exact right size.
    std::uint8_t xPresentBuf[256];
    std::uint8_t yPresentBuf[256];
    std::uint8_t stackBuf[256];

    CKHash x = fillCKHash(X, xIndices, xPresentBuf);
    CKHash y = fillCKHash(Y, yIndices, yPresentBuf);

    double Sa = 0.0, Sb = 0.0, D = 0.0;
    std::size_t stackTop = 0;

    auto matchOne = [&](std::span<const std::uint8_t> xSpan, std::span<const std::uint8_t> ySpan) {
        bool swapped = false;
        auto shortSpan = xSpan, longSpan = ySpan;
        if (shortSpan.size() > longSpan.size()) {
            std::swap(shortSpan, longSpan);
            swapped = true;
        }

        std::size_t i = 0, j = 0;
        const std::size_t longSize = longSpan.size();
        stackTop = 0;

        while (i < shortSpan.size()) {
            const std::uint8_t xhead = shortSpan[i];
            ++i;

            while (j < longSize && longSpan[j] < xhead) {
                stackBuf[stackTop++] = longSpan[j];
                ++j;
            }

            const bool sEmpty = (stackTop == 0);
            const bool qyEmpty = (j >= longSize);

            if (!sEmpty && !qyEmpty) {
                const int left = static_cast<int>(xhead) - static_cast<int>(stackBuf[stackTop - 1]);
                const int right = static_cast<int>(longSpan[j]) - static_cast<int>(xhead);
                if (left < right) {
                    D += static_cast<double>(left);
                    --stackTop;
                } else {
                    D += static_cast<double>(right);
                    ++j;
                }
            } else if (!sEmpty) {
                D += static_cast<double>(xhead - stackBuf[stackTop - 1]);
                --stackTop;
            } else if (!qyEmpty) {
                D += static_cast<double>(longSpan[j] - xhead);
                ++j;
            }
        }

        const double leftover = static_cast<double>(stackTop) + static_cast<double>(longSize - j);
        if (swapped) {
            Sa += leftover;
        } else {
            Sb += leftover;
        }
    };

    for (std::size_t k = 0; k < x.presentCount; ++k) {
        const unsigned char c = x.present[k];
        // One y[c] lookup instead of y.contains(c) followed by a separate
        // y[c] -- both independently recompute sparse[c]; an empty span
        // (only possible when c isn't present, since a present character
        // always has count >= 1) already answers both questions at once.
        auto ySpan = y[c];
        if (!ySpan.empty()) {
            matchOne(x.spanAt(k), ySpan);
        } else {
            Sa += static_cast<double>(x.spanAt(k).size());
        }
    }
    for (std::size_t k = 0; k < y.presentCount; ++k) {
        const unsigned char c = y.present[k];
        if (!x.contains(c)) Sb += static_cast<double>(y.spanAt(k).size());
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

// Falls back to ck::sparse (uint32_t, arena-backed) whenever either string
// doesn't fit the byte-width precondition, rather than exposing that split
// to callers.
inline double ckSimilarity(const std::string& A, const std::string& B) {
    if (A.size() >= kMaxLen || B.size() >= kMaxLen) {
        return ck::sparse::ckSimilarity(A, B);
    }

    const bool aFirst = (A.size() != B.size()) ? (A.size() < B.size()) : (A <= B);
    const std::string& X = aFirst ? A : B;
    const std::string& Y = aFirst ? B : A;

    return ckSimilarityFromMeasures(ckMeasuresSparse8(X, Y));
}

} // namespace ck::sparse8
