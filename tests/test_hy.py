"""Hayashi-Yoshida acceptance gate (Week 4): 4 synthetic recipes + parity.

The first test DEFINES the sign convention (Y follows X by delta -> peak at -delta).
numba<->numpy parity runs always; C++ parity is importorskip'd so CI is green
without the compiled extension (the fallback path is the tested default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxpd.econo import hy
from fxpd.econo import _hy_numpy as HNP
from fxpd.econo import _hy_numba as HNB

NS = 1_000_000_000
FIX = Path(__file__).parent / "fixtures"


def _poisson_rw(T, lam, seed, sigma=1e-3):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, T, rng.poisson(lam * T)))
    t_ns = (t * NS).astype(np.int64)
    x = np.concatenate([[0.0], np.cumsum(rng.normal(0, sigma, len(t_ns) - 1))])
    return t_ns, x, rng


def _locf(t_ns, x, s_ns):
    idx = np.searchsorted(t_ns, s_ns, side="right") - 1
    return np.where(idx < 0, 0.0, x[np.clip(idx, 0, len(x) - 1)])


def _fixture_arrays(sym):
    from fxpd.data.clean import clean_ticks
    from fxpd.data.mids import logmid_series

    df = clean_ticks(pd.read_parquet(FIX / f"{sym}-2026-06-18-0900_1000.parquet")).ticks
    p = logmid_series(df)
    return p.index.view("i8").astype(np.int64), p.to_numpy(float)


# --- 1. Known lead -> DEFINES the sign convention ---------------------------------
def test_known_lead_defines_sign_convention():
    T, delta = 2000.0, int(0.200 * NS)
    tx, x, rng = _poisson_rw(T, 5, 1)
    ty = np.sort(rng.uniform(0, T, rng.poisson(5 * T)))
    ty_ns = (ty * NS).astype(np.int64)
    y = _locf(tx, x, ty_ns - delta) + rng.normal(0, 1e-5, len(ty_ns))  # Y(t)=X(t-200ms)
    thetas = hy.theta_grid_ns()
    s = hy.leadlag_summary(thetas, hy.hy_leadlag(tx, x, ty_ns, y, thetas))
    assert s["theta_star_ms"] == -200.0     # X (first arg) leads by 200 ms
    assert abs(s["rho_peak"]) > 0.8
    assert s["asymmetry_S"] > 0


# --- 2. Independent random walks -> no lead ---------------------------------------
def test_independent_rw_no_lead():
    tx, x, _ = _poisson_rw(2000.0, 5, 2)
    ty, y, _ = _poisson_rw(2000.0, 5, 3)
    thetas = hy.theta_grid_ns()
    rho = hy.hy_leadlag(tx, x, ty, y, thetas)
    assert np.max(np.abs(rho)) < 3 / np.sqrt(min(len(x), len(y)))


# --- 3. Contemporaneous correlation, no lead -> peak at 0, symmetric ---------------
def test_contemporaneous_peak_at_zero():
    T, N = 2000.0, 40_000
    z = np.cumsum(np.random.default_rng(4).normal(0, 1e-3, N))
    tt = np.linspace(0, T, N)

    def sample(seed):
        r = np.random.default_rng(seed)
        s = np.sort(r.uniform(0, T, r.poisson(5 * T)))
        idx = np.searchsorted(tt, s, side="right") - 1
        val = z[np.clip(idx, 0, N - 1)] + r.normal(0, 1e-5, len(s))
        return (s * NS).astype(np.int64), val

    tx, x = sample(5)
    ty, y = sample(6)
    thetas = hy.theta_grid_ns()
    s = hy.leadlag_summary(thetas, hy.hy_leadlag(tx, x, ty, y, thetas))
    # No directional lead: symmetric curve (S ~ 0, the robust statistic) and a peak
    # within the tens-of-ms resolution floor of 0 (argmax can jitter to +/-1-2 ms on
    # a fine grid under finite-sample noise -- that is "no resolvable lead").
    assert abs(s["asymmetry_S"]) < 0.05
    assert abs(s["theta_star_ms"]) <= 10.0


# --- 4. HY == realized covariance on a synchronous grid (exactness) ----------------
def test_hy_equals_realized_cov_on_sync_grid():
    rng = np.random.default_rng(0)
    t = (np.arange(5000) * 1_000_000).astype(np.int64)
    dx = rng.normal(0, 1e-4, 4999)
    dy = 0.5 * dx + rng.normal(0, 1e-4, 4999)
    x = np.concatenate([[0.0], np.cumsum(dx)])
    y = np.concatenate([[0.0], np.cumsum(dy)])
    assert abs(hy.hy_cov(t, x, t, y) - float(np.sum(np.diff(x) * np.diff(y)))) < 1e-14


# --- Parity: numba vs numpy (synthetic + real fixture) ----------------------------
@pytest.mark.parametrize("source", ["synthetic", "fixture"])
def test_parity_numba_vs_numpy(source):
    if source == "synthetic":
        tx, x, _ = _poisson_rw(500.0, 5, 7)
        ty, y, _ = _poisson_rw(500.0, 5, 8)
    else:
        tx, x = _fixture_arrays("EURUSD")
        ty, y = _fixture_arrays("EURJPY")
    thetas = hy.theta_grid_ns()
    assert abs(HNP.hy_cov(tx, x, ty, y) - HNB.hy_cov(tx, x, ty, y)) < 1e-12
    r_np = HNP.hy_leadlag(tx, x, ty, y, thetas)
    r_nb = HNB.hy_leadlag(tx, x, ty, y, thetas)
    assert np.max(np.abs(r_np - r_nb)) < 1e-12


# --- Parity: C++ vs numpy (skipped if fxpd_fast not built -> CI green uncompiled) --
def test_parity_cpp_vs_numpy_real_fixture():
    fast = pytest.importorskip("fxpd_fast")
    tx, x = _fixture_arrays("EURUSD")
    ty, y = _fixture_arrays("EURJPY")
    thetas = hy.theta_grid_ns()
    assert abs(fast.hy_cov(tx, x, ty, y) - HNP.hy_cov(tx, x, ty, y)) < 1e-12
    r_cpp = fast.hy_leadlag(tx, x, ty, y, thetas)
    assert np.max(np.abs(r_cpp - HNP.hy_leadlag(tx, x, ty, y, thetas))) < 1e-12


def test_driver_leadlag_windows(tmp_path, monkeypatch):
    """Exercise the data-layer driver path (_instr_arrays/_series_arrays/windows)."""
    import datetime as dt
    import fxpd.data.store as store
    from fxpd.config import TRIANGLES

    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    tri = TRIANGLES["majors"]
    date = dt.date(2026, 6, 15)
    ts = pd.date_range(f"{date} 08:00", f"{date} 12:00", freq="1s", tz="UTC")
    rng = np.random.default_rng(0)
    for sym, base in [("EURJPY", 160.0), ("EURUSD", 1.10), ("USDJPY", 145.0)]:
        mid = base + np.cumsum(rng.normal(0, base * 1e-6, len(ts)))
        store.write_day(sym, date, pd.DataFrame({
            "ts": ts, "bid": mid - base * 1e-5, "ask": mid + base * 1e-5,
            "bid_volume": 1.0, "ask_volume": 1.0}))
    out = hy.leadlag_windows(tri, [date], "cross", "synth", liquid=(7, 17), min_ticks=50)
    assert len(out) == 1 and out.attrs["skipped"] == 0
    assert np.isfinite(out["theta_star_ms"].iloc[0])
    assert out.attrs["rho"].shape == (1, len(hy.theta_grid_ns()))


def test_backend_and_smoke():
    assert hy.BACKEND in {"cpp", "numba", "numpy"}
    tx, x, _ = _poisson_rw(200.0, 5, 9)
    ty, y, _ = _poisson_rw(200.0, 5, 10)
    thetas = hy.theta_grid_ns()
    rho = hy.hy_leadlag(tx, x, ty, y, thetas)
    assert rho.shape == thetas.shape and np.all(np.isfinite(rho))
