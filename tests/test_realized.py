"""Realized kernel (Week 6). The noise-robustness contrast IS pitfall #6 as a test."""

from __future__ import annotations

import numpy as np

from fxpd.vol.realized import (
    bandwidth_bnhls,
    noise_variance,
    parzen,
    realized_kernel,
)


def test_parzen_shape():
    assert parzen(np.array([0.0]))[0] == 1.0
    assert parzen(np.array([1.0]))[0] == 0.0
    assert parzen(np.array([1.5]))[0] == 0.0
    # continuous at 1/2 (both branches meet at 1/4)
    left, right = parzen(np.array([0.5 - 1e-9]))[0], parzen(np.array([0.5 + 1e-9]))[0]
    assert abs(left - right) < 1e-6
    assert np.all(parzen(np.linspace(0, 1, 50)) >= 0)


def test_rk_recovers_iv_while_naive_rv_explodes():
    """Diffusion + iid microstructure noise with known IV=1.

    Asserted on the MEAN over paths, not a single path: measured single-path error is
    median ~10% / p90 ~22%, so a one-path 15% assertion would be flaky ~1 in 3
    (WEEK6.md finding). The estimator's claim is unbiasedness -- test that.
    """
    rng = np.random.default_rng(0)
    n, sigma, omega, IV, reps = 2000, 1.0, 0.05, 1.0, 120
    rk_s, rv_s = [], []
    for _ in range(reps):
        d_true = sigma * np.sqrt(1.0 / n) * rng.normal(size=n)
        eps = omega * rng.normal(size=n + 1)
        r = d_true + (eps[1:] - eps[:-1])
        H = bandwidth_bnhls(r, iq=IV**2)
        rk_s.append(realized_kernel(r, H))
        rv_s.append(float(np.sum(r * r)))
    rk_mean, rv_mean = float(np.mean(rk_s)), float(np.mean(rv_s))
    assert abs(rk_mean / IV - 1) < 0.15, f"RK mean off: {rk_mean}"
    assert rv_mean / IV > 3.0, f"naive RV should explode, got {rv_mean}"


def test_noise_variance_floors_at_zero_for_positive_autocov():
    """omega^2 floors at 0 when returns have POSITIVE first-order autocovariance.

    This is exactly what happens on cleaned log-MID returns (the mid removes the
    bid-ask bounce that the estimator is designed to detect), which is why the
    kernel correction is near-inert on our data -- documented, not hidden.
    """
    trending = np.array([1.0, 1.0, 1.0, 1.0, 1.0])       # gamma_1 > 0
    assert noise_variance(trending) == 0.0
    bouncing = np.array([1.0, -1.0, 1.0, -1.0, 1.0])     # gamma_1 < 0 -> real noise
    assert noise_variance(bouncing) > 0.0


def test_bandwidth_degenerates_without_detectable_noise():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1e-4, 1000)          # no bounce -> omega^2 ~ 0
    assert bandwidth_bnhls(r, iq=1e-8) == 1
    # with real bid-ask-bounce-like noise the bandwidth opens up
    eps = 0.05 * rng.normal(size=1001)
    r_noisy = rng.normal(0, 1 / np.sqrt(1000), 1000) + (eps[1:] - eps[:-1])
    assert bandwidth_bnhls(r_noisy, iq=1.0) > 1
