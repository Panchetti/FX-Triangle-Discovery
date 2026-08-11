"""Synthetic-recovery test for fxpd.econo.vecm -- the Week-2 acceptance gate.

Known DGP (WEEK2.md recipe): two independent random-walk efficient prices
w1, w2; legs observe them with tiny noise; the cross carries an AR(1)
mispricing u with phi = 0.98, so ONLY the cross error-corrects and the true
loadings for the z-regression are alpha = (phi - 1, 0, 0) = (-0.02, 0, 0),
with half-life ln 2 / (1 - phi) ~ 34.66 bars. This test PINS every sign
convention in vecm.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from fxpd.config import TRIANGLES
from fxpd.econo.vecm import (
    BIVARIATE,
    TRIVARIATE,
    estimate_window,
    frame_names,
    half_life_seconds,
    make_frame,
)

TRI = TRIANGLES["majors"]
PHI = 0.98
T = 50_000
TRUE_HL_BARS = np.log(2) / (1 - PHI)  # ~34.66


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    w1 = rng.normal(0, 1e-4, T).cumsum()
    w2 = rng.normal(0, 1e-4, T).cumsum()
    u = lfilter([1.0], [1.0, -PHI], rng.normal(0, 0.3e-4, T))  # AR(1), phi=0.98
    leg1 = w1 + 0.05e-4 * rng.normal(size=T)
    leg2 = w2 + 0.05e-4 * rng.normal(size=T)
    cross = w1 + w2 + u
    idx = pd.date_range("2026-06-15", periods=T, freq="1s", tz="UTC")
    return pd.DataFrame(
        {TRI.cross: cross, TRI.leg1: leg1, TRI.leg2: leg2}, index=idx
    )


def _fit(panel, frame, k=2):
    x = make_frame(panel, TRI, frame)
    return estimate_window(x, k, frame, frame_names(TRI, frame), bar_seconds=1.0)


def test_trivariate_recovers_known_alpha(panel):
    r = _fit(panel, TRIVARIATE)
    # only the cross error-corrects: alpha ~ (phi-1, 0, 0)
    assert abs(r.alpha[0] - (PHI - 1)) < 3 * r.alpha_se[0]
    # Strongly significant. NB: under this DGP the theoretical plain-OLS t is
    # ~ -4.7 (= -0.02 * sd(z) * sqrt(T) / sd(resid)), so a -5 threshold is
    # unattainable by construction; -3 asserts significance without flakiness.
    assert r.alpha_t[0] < -3
    assert abs(r.alpha[1]) < 3 * r.alpha_se[1]  # legs insignificant
    assert abs(r.alpha[2]) < 3 * r.alpha_se[2]
    # half-life within 20% of ln2/(1-phi) bars (bar_seconds=1)
    assert r.half_life_s == pytest.approx(TRUE_HL_BARS, rel=0.20)
    assert -1 < r.beta_alpha < 0


def test_bivariate_same_data(panel):
    r = _fit(panel, BIVARIATE)
    assert r.names == [TRI.cross, "synth"]
    assert abs(r.alpha[0] - (PHI - 1)) < 3 * r.alpha_se[0]  # alpha_c ~ phi-1
    assert abs(r.alpha[1]) < 3 * r.alpha_se[1]              # alpha_s ~ 0
    assert r.half_life_s == pytest.approx(TRUE_HL_BARS, rel=0.20)


def test_ordering_comes_from_triangle_not_caller(panel):
    """make_frame must select by name; a shuffled panel changes nothing."""
    shuffled = panel[[TRI.leg1, TRI.cross, TRI.leg2]]
    x_ref = make_frame(panel, TRI, TRIVARIATE)
    x_shuf = make_frame(shuffled, TRI, TRIVARIATE)
    assert list(x_shuf.columns) == [TRI.cross, TRI.leg1, TRI.leg2, "z"]
    pd.testing.assert_frame_equal(x_ref, x_shuf)


def test_half_life_pure_function():
    assert half_life_seconds(-0.02, 1.0) == pytest.approx(TRUE_HL_BARS, rel=0.02)
    assert np.isnan(half_life_seconds(+0.01, 1.0))   # explosive -> NaN
    assert np.isnan(half_life_seconds(-1.5, 1.0))    # overshooting -> NaN
    # bar length scales linearly
    assert half_life_seconds(-0.02, 5.0) == pytest.approx(
        5 * half_life_seconds(-0.02, 1.0)
    )


def test_dynamic_half_life_matches_proxy_only_when_gamma_is_zero():
    """The beta'alpha proxy matches the dynamic first-passage within interpolation
    error (< 1 bar) ONLY for Gamma=0; with lags they diverge materially (audit finding C1).

    NOTE (audit-3 #7): even at Gamma=0 the interpolated first-passage is not *exactly*
    the analytic AR(1) half-life -- e.g. for phi=0.8 analytic ln2/(-ln.8)=3.106 vs linear
    interpolation ~3.117 -- so this asserts agreement to within one bar, NOT equality.
    The exact decoupling condition is beta'Gamma_i=0 for all i (which permits nonzero
    Gamma_i), not Gamma=0; Gamma=0 is the simplest sufficient case tested here.

    The decay errs in BOTH directions once beta'Gamma_i != 0. NB (audit-4 #7): the
    direction is a property of the realized sequence beta'Gamma_i dx_{t-i} and the
    companion eigenstructure, NOT a general "sign of Gamma" law -- the two diagonal Gammas
    below happen to speed/slow it, but that is these specific matrices, not a theorem.
    Encoded so the headline speed metric can never again be called "the half-life".
    """
    from fxpd.econo.vecm import dynamic_half_life

    beta = np.array([1.0, -1.0])
    alpha = np.array([-0.05, 0.05])          # beta'alpha = -0.10
    proxy = half_life_seconds(float(alpha[0] - alpha[1]), 1.0)

    # Gamma = 0 -> the scalar recursion IS the system; agree to within one bar
    dyn0 = dynamic_half_life(alpha, np.zeros((2, 2, 2)), beta, 1.0)
    assert abs(dyn0 - proxy) <= 1.0

    # these SPECIFIC positive-diagonal Gammas speed it up; negative-diagonal slow it
    g_pos = np.stack([np.eye(2) * 0.4, np.eye(2) * 0.2])
    assert dynamic_half_life(alpha, g_pos, beta, 1.0) < proxy - 1.0
    g_neg = np.stack([np.eye(2) * -0.4, np.eye(2) * -0.2])
    assert dynamic_half_life(alpha, g_neg, beta, 1.0) > proxy + 1.0


def test_dynamic_half_life_diagnostics():
    """return_diagnostics exposes rebound/censoring so their RATES are reportable (audit-4 #3)."""
    from fxpd.econo.vecm import dynamic_half_life

    beta = np.array([1.0, -1.0])
    # monotone decay: crosses cleanly, never rebounds above 1/2
    hl, d = dynamic_half_life(np.array([-0.1, 0.1]), np.zeros((1, 2, 2)), beta, 1.0,
                              return_diagnostics=True)
    assert d["crossed"] and not d["censored"] and not d["rebound"]
    assert d["bars_to_cross"] >= 1 and np.isfinite(hl)
    # explosive: beta'alpha = +0.2 (phi=1.2) grows without bound -> censored within max_h
    hl2, d2 = dynamic_half_life(np.array([0.1, -0.1]), np.zeros((1, 2, 2)), beta, 1.0,
                                max_h=50, return_diagnostics=True)
    assert d2["censored"] and not d2["crossed"] and np.isnan(hl2)
    # diagnostics path and fast path agree on the half-life
    assert dynamic_half_life(np.array([-0.1, 0.1]), np.zeros((1, 2, 2)), beta, 1.0) == hl
