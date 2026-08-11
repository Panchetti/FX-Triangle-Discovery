"""Acceptance gate for fxpd.econo.discovery (Week 3).

Two kinds of test:
- pure-function exactness (Cholesky bookkeeping) on hand-built Omega, where the
  playbook's <1e-10 tolerances actually hold;
- simulate -> fit_system -> discovery on known DGPs (known-leader recovery, and
  the Psi(1) proportionality cross-check that pins the Gamma orientation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxpd.econo import discovery as disc
from fxpd.econo import vecm

NAMES = ["cross", "synth"]


def _sim(alpha, T=60_000, gamma1=None, sigma=1e-4, seed=0) -> pd.DataFrame:
    """Bivariate VECM DGP. Δp_t = alpha*(cross-synth)_{t-1} + gamma1·Δp_{t-1} + eps."""
    rng = np.random.default_rng(seed)
    alpha = np.asarray(alpha, float)
    p = np.zeros((T, 2))
    eps = rng.normal(0, sigma, (T, 2))
    for t in range(2, T):
        z = p[t - 1, 0] - p[t - 1, 1]
        dp = alpha * z + eps[t]
        if gamma1 is not None:
            dp = dp + gamma1 @ (p[t - 1] - p[t - 2])
        p[t] = p[t - 1] + dp
    idx = pd.date_range("2026-06-15", periods=T, freq="1s", tz="UTC")
    x = pd.DataFrame({"cross": p[:, 0], "synth": p[:, 1]}, index=idx)
    x["z"] = x["cross"] - x["synth"]
    return x


def _prop_err(M: np.ndarray, ap: np.ndarray) -> float:
    """Max |sin(angle)| between each row of M and alpha_perp (0 == exactly parallel)."""
    return max(abs(np.cross(row, ap)) / (np.linalg.norm(row) * np.linalg.norm(ap)) for row in M)


# --------------------------------------------------------------------------- #
# 1. Known leader: cross leads (alpha_c ~ 0), synth error-corrects.
# --------------------------------------------------------------------------- #
def test_known_leader():
    # Cross is ~96% the leader: it corrects only weakly (alpha_c small negative),
    # synth strongly. True GG_cross = 0.10/0.105 = 0.952 -- inside [0,1] with room,
    # so the window is valid (an alpha_c of exactly 0 lands GG on the >1 boundary).
    x = _sim(alpha=[-0.005, 0.10], T=100_000, seed=1)
    sys = vecm.fit_system(x, k=2, names=NAMES)
    r = disc.discovery_window(sys, bar_seconds=1.0)
    assert r.valid
    assert r.gg_cross > 0.95
    assert r.is_cross_lo > 0.85


# --------------------------------------------------------------------------- #
# 2. Diagonal Omega -> the two Cholesky orderings coincide (exact).
# --------------------------------------------------------------------------- #
def test_is_bounds_diagonal_omega_zero_width():
    alpha = np.array([-0.05, 0.15])
    is_ = disc.information_shares(alpha, np.diag([2e-8, 5e-9]))
    assert abs(is_["cross"][1] - is_["cross"][0]) < 1e-10
    assert abs(is_["synth"][1] - is_["synth"][0]) < 1e-10
    assert abs(is_["resid_corr"]) < 1e-12


def test_is_ordering_sums_to_one():
    alpha = np.array([-0.05, 0.15])
    omega = np.array([[1e-8, 0.6e-8], [0.6e-8, 1e-8]])
    psi = disc.alpha_perp(alpha)
    den = psi @ omega @ psi
    L = np.linalg.cholesky(omega)
    assert ((psi @ L) ** 2 / den).sum() == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. Symmetric system -> GG ~ 0.5, IS band straddles 0.5.
# --------------------------------------------------------------------------- #
def test_symmetric_5050():
    alpha = np.array([-0.1, 0.1])
    assert abs(disc.gg_shares(alpha)["cross"] - 0.5) < 0.05
    is_ = disc.information_shares(alpha, np.array([[1e-8, 0.5e-8], [0.5e-8, 1e-8]]))
    lo, hi = is_["cross"]
    assert lo < 0.5 < hi


# --------------------------------------------------------------------------- #
# 4. Psi(1) proportionality: companion iteration and closed form both give rows
#    proportional to alpha_perp, on three fitted DGPs. Pins Gamma orientation.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "alpha, gamma1, seed",
    [
        ([0.0, 0.05], None, 1),                       # leader
        ([-0.08, 0.08], None, 2),                     # symmetric
        ([-0.05, 0.10], np.array([[0.2, 0.0],
                                  [0.0, 0.3]]), 3),   # with lag dynamics (nonzero Gamma)
    ],
)
def test_psi1_proportional_to_alpha_perp(alpha, gamma1, seed):
    x = _sim(alpha=alpha, T=60_000, gamma1=gamma1, seed=seed)
    sys = vecm.fit_system(x, k=2, names=NAMES)
    ap = disc.alpha_perp(sys.alpha)
    psi_c = disc.psi_companion(sys.alpha, sys.gamma)
    psi_cf = disc.long_run_impact(sys.alpha, sys.gamma)
    assert _prop_err(psi_c, ap) < 1e-6
    assert _prop_err(psi_cf, ap) < 1e-6


# --------------------------------------------------------------------------- #
# Pure-function guards + fit_system parity (finding #6).
# --------------------------------------------------------------------------- #
def test_gg_nan_guard_and_sign_flip():
    assert np.isnan(disc.gg_shares(np.array([0.0, 0.0]))["cross"])
    # sign-flipped alpha_c > 0 -> GG > 1 returned as-is (caller's range guard excludes)
    assert disc.gg_shares(np.array([0.01, 0.15]))["cross"] > 1.0
    a = np.array([-0.05, 0.15])
    assert disc.alpha_perp(a) @ a == pytest.approx(0.0, abs=1e-18)


def test_discovery_window_excludes_out_of_range_gg():
    # alpha_c flipped positive -> GG_cross > 1 -> valid must be False
    x = _sim(alpha=[0.0, 0.05], T=40_000, seed=7)
    sys = vecm.fit_system(x, k=2, names=NAMES)
    sys.alpha[0] = 0.02  # force a sign flip so GG_cross > 1
    r = disc.discovery_window(sys, bar_seconds=1.0)
    assert r.gg_cross > 1.0 and r.valid is False


def test_fit_system_alpha_matches_estimate_window():
    x = _sim(alpha=[-0.05, 0.12], T=30_000, seed=4)
    a_sys = vecm.fit_system(x, k=2, names=NAMES).alpha
    a_ew = vecm.estimate_window(x, 2, vecm.BIVARIATE, NAMES, bar_seconds=1.0).alpha
    assert np.allclose(a_sys, a_ew, atol=1e-10)
