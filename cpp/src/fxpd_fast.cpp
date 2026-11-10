// fxpd_fast — Hayashi-Yoshida covariation + lead-lag hot loop (Week 4).
//
// Optional benchmark port of the O(n+m) two-pointer in fxpd.econo.hy (numba), which the
// library actually uses -- numba measured faster than this module. Parity to the numba
// version is asserted to < 1e-12 in tests/test_hy.py. The lead-lag shift is applied as
// an int64 offset inside the interval comparison — no array copies.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>

namespace py = pybind11;

using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Covariation with Y's timestamps shifted by `theta` (nanoseconds).
static double hy_cov_shifted(const I64 &tx_, const F64 &x_, const I64 &ty_,
                             const F64 &y_, int64_t theta) {
    auto tx = tx_.unchecked<1>();
    auto x = x_.unchecked<1>();
    auto ty = ty_.unchecked<1>();
    auto y = y_.unchecked<1>();
    const py::ssize_t n = tx.shape(0) - 1;  // number of X intervals
    const py::ssize_t m = ty.shape(0) - 1;  // number of Y intervals
    py::ssize_t i = 0, j = 0;
    double cov = 0.0;
    while (i < n && j < m) {
        const int64_t axi = tx(i), bxi = tx(i + 1);
        const int64_t ayj = ty(j) + theta, byj = ty(j + 1) + theta;
        if (bxi > ayj && byj > axi)          // (axi,bxi] and (ayj,byj] overlap
            cov += (x(i + 1) - x(i)) * (y(j + 1) - y(j));
        if (bxi <= byj)                      // advance the interval that ends first
            ++i;
        else
            ++j;
    }
    return cov;
}

static double hy_cov(I64 tx, F64 x, I64 ty, F64 y) {
    return hy_cov_shifted(tx, x, ty, y, 0);
}

static F64 hy_leadlag(I64 tx, F64 x, I64 ty, F64 y, I64 thetas_) {
    auto x_ = x.unchecked<1>();
    auto y_ = y.unchecked<1>();
    const py::ssize_t n = x_.shape(0) - 1, m = y_.shape(0) - 1;
    double rvx = 0.0, rvy = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        const double d = x_(i + 1) - x_(i);
        rvx += d * d;
    }
    for (py::ssize_t j = 0; j < m; ++j) {
        const double d = y_(j + 1) - y_(j);
        rvy += d * d;
    }
    const double denom = std::sqrt(rvx * rvy);
    auto th = thetas_.unchecked<1>();
    const py::ssize_t K = th.shape(0);
    F64 out(K);
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t k = 0; k < K; ++k)
        o(k) = hy_cov_shifted(tx, x, ty, y, th(k)) / denom;
    return out;
}

PYBIND11_MODULE(fxpd_fast, m) {
    m.doc() = "Hayashi-Yoshida covariation + lead-lag (C++ hot loop)";
    m.def("hy_cov", &hy_cov, py::arg("tx"), py::arg("x"), py::arg("ty"), py::arg("y"));
    m.def("hy_leadlag", &hy_leadlag, py::arg("tx"), py::arg("x"), py::arg("ty"),
          py::arg("y"), py::arg("thetas_ns"));
}
