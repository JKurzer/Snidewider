#pragma once
// Minimal stand-in so L0Learn's core compiles without R/Rcpp.
// The only live Rcpp symbol the core uses is checkUserInterrupt().
// <numeric> is for std::iota (MSVC doesn't pull it in transitively).
#include <numeric>

#include <armadillo>

namespace Rcpp {
inline void checkUserInterrupt() {}
}  // namespace Rcpp
