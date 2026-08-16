// pybind11 bindings for the vendored L0Learn core (L0/L0L1/L0L2 logistic).
// Bypasses the Rcpp Interface layer: drives Grid<> directly. Small problems
// only (ours are ~6000 x ~90) — dense arma::mat, column-major wrap from numpy.
#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <string>
#include <vector>

#include "include/Grid.h"

namespace py = pybind11;

namespace {

py::dict fit(py::array_t<double, py::array::f_style> X, py::array_t<double> y,
             const std::string& penalty, double gamma, std::size_t n_lambda,
             std::size_t max_nnz, std::size_t max_iters) {
    const auto Xb = X.unchecked<2>();
    const std::size_t n = Xb.shape(0), p = Xb.shape(1);
    arma::mat Xm(const_cast<double*>(Xb.data(0, 0)), n, p, true);  // copy (f_style)
    const auto yb = y.unchecked<1>();
    arma::vec yv(const_cast<double*>(yb.data(0)), n, true);
    // L0Learn's logistic loss is log(1+exp(-y*f)); labels must be +-1.
    // Accept any sign convention here: strictly positive -> +1, else -1.
    yv.transform([](double v) { return v > 0.0 ? 1.0 : -1.0; });

    GridParams<arma::mat> PG;
    PG.P.Specs.Logistic = true;
    PG.P.Specs.Classification = true;
    PG.P.Specs.CD = true;
    if (penalty == "L0") {
        PG.P.Specs.L0 = true;
    } else if (penalty == "L0L1") {
        PG.P.Specs.L0L1 = true;
        PG.P.ModelParams[1] = gamma;
    } else if (penalty == "L0L2") {
        PG.P.Specs.L0L2 = true;
        PG.P.ModelParams[2] = gamma;
    } else {
        throw std::invalid_argument("penalty must be L0, L0L1, or L0L2");
    }
    PG.G_ncols = n_lambda;
    PG.G_nrows = 5;
    PG.NnzStopNum = max_nnz;
    PG.intercept = true;
    PG.P.intercept = true;
    PG.P.MaxIters = max_iters;

    // P.Xy (y .* X on the normalized design) is wired up inside Grid::Fit
    // for the pure-L0 path; Grid2D builds its own for L0L1/L0L2.
    Grid<arma::mat> G(Xm, yv, PG);
    G.Fit();

    // L0 => single 1D path in row 0; L0L1/L0L2 => 2D grid rows = gamma values.
    py::dict out;
    py::list lambdas, nnz, intercepts, converged, gammas;
    py::list betas;
    for (std::size_t i = 0; i < G.Solutions.size(); ++i) {
        const std::size_t path_len = G.Solutions[i].size();
        py::array_t<double> B(std::vector<std::ptrdiff_t>{(std::ptrdiff_t)p, (std::ptrdiff_t)path_len});
        auto Bm = B.mutable_unchecked<2>();
        for (std::size_t j = 0; j < path_len; ++j) {
            arma::mat dense(G.Solutions[i][j]);
            for (std::size_t r = 0; r < p; ++r) Bm(r, j) = dense(r, 0);
            // NB: Converged is std::vector<bool>; its proxy reference casts
            // to a null py::object. Explicit py::bool_ or PyList_Append AVs.
            converged.append(py::bool_(G.Converged[i][j]));
            lambdas.append(G.Lambda0[i][j]);
            nnz.append(G.NnzCount[i][j]);
            intercepts.append(G.Intercepts[i][j]);
            gammas.append(G.Lambda12[i]);
        }
        betas.append(B);  // one matrix per gamma row (single row for pure L0)
    }
    out["betas"] = betas;
    out["lambdas"] = lambdas;
    out["gammas"] = gammas;
    out["nnz"] = nnz;
    out["intercepts"] = intercepts;
    out["converged"] = converged;
    return out;
}

}  // namespace

PYBIND11_MODULE(_l0learn_native, m) {
    m.doc() = "L0Learn core (vendored, hazimehh/L0Learn) — best-subset logistic paths.";
    m.def("fit", &fit, py::arg("X"), py::arg("y"), py::arg("penalty") = "L0",
          py::arg("gamma") = 0.001, py::arg("n_lambda") = 50, py::arg("max_nnz") = 20,
          py::arg("max_iters") = 500,
          "Fit an L0/L0L1/L0L2 logistic path. X must be Fortran-order float64.");
}
