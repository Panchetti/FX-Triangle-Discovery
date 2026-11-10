"""Hayashi-Yoshida lead-lag on raw asynchronous ticks."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from numba import njit

from ..config import Triangle

# Single implementation (numba). An optional pybind11 port lives in cpp/ purely as a
# benchmark: it was measured SLOWER than this (19 ms vs 12 ms on one liquid day), so the
# library never imports it. The pure-Python reference is `hy_cov.py_func`.

NS_PER_MS = 1_000_000
THETAS_MS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


# error_model="numpy": a zero realized variance gives nan/inf (as the C++ port does), not a
# ZeroDivisionError that would abort a whole leadlag_windows loop on one flat series.
@njit(cache=True, error_model="numpy")
def hy_cov(tx, x, ty, y):
    """Hayashi-Yoshida covariation over half-open interval overlaps, O(n+m) two-pointer.

    Returns and interval endpoints are precomputed as contiguous arrays: indexing i+1 inside
    the loop instead measured 1.8x slower under numba (22 ms vs 12 ms on one liquid day).
    """
    rx = x[1:] - x[:-1]
    ry = y[1:] - y[:-1]
    ax, bx = tx[:-1], tx[1:]
    ay, by = ty[:-1], ty[1:]
    i = 0
    j = 0
    n = rx.shape[0]
    m = ry.shape[0]
    cov = 0.0
    while i < n and j < m:
        if bx[i] > ay[j] and by[j] > ax[i]:               # (ax_i, bx_i] and (ay_j, by_j] overlap
            cov += rx[i] * ry[j]
        if bx[i] <= by[j]:                                 # advance the interval that ends first
            i += 1
        else:
            j += 1
    return cov


@njit(cache=True, error_model="numpy")
def hy_leadlag(tx, x, ty, y, thetas_ns):
    """rho(theta) = HY-corr(X, Y shifted by +theta) for each theta (nanoseconds)."""
    rvx = np.sum((x[1:] - x[:-1]) ** 2)
    rvy = np.sum((y[1:] - y[:-1]) ** 2)
    denom = np.sqrt(rvx * rvy)
    out = np.empty(thetas_ns.shape[0], dtype=np.float64)
    for k in range(thetas_ns.shape[0]):
        out[k] = hy_cov(tx, x, ty + thetas_ns[k], y) / denom
    return out


def theta_grid_ns() -> np.ndarray:
    """Symmetric lead-lag grid -/+ THETAS_MS plus 0, in int64 nanoseconds."""
    pos = np.array(THETAS_MS, dtype=np.int64)
    return np.concatenate([-pos[::-1], [0], pos]).astype(np.int64) * NS_PER_MS

def leadlag_summary(thetas_ns: np.ndarray, rho: np.ndarray) -> dict:
    """theta* = argmax|rho|, plus the peak rho and the asymmetry statistic S."""
    thetas_ms = thetas_ns / NS_PER_MS
    k = int(np.argmax(np.abs(rho)))
    ar = np.abs(rho)
    neg, pos = ar[thetas_ns < 0], ar[thetas_ns > 0]
    S = (neg.max() if neg.size else 0.0) - (pos.max() if pos.size else 0.0)
    return {"theta_star_ms": float(thetas_ms[k]), "rho_peak": float(rho[k]),
            "asymmetry_S": float(S)}

def synth_ticks(
    t1: np.ndarray, p1: np.ndarray, t2: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Async synthetic-cross ticks: union event times, LOCF sum of the two log mids."""
    ev = np.union1d(t1, t2)
    i1 = np.searchsorted(t1, ev, side="right") - 1
    i2 = np.searchsorted(t2, ev, side="right") - 1
    ok = (i1 >= 0) & (i2 >= 0)
    return ev[ok], p1[i1[ok]] + p2[i2[ok]]

def _instr_arrays(instrument: str, date: dt.date, liquid: tuple[int, int]):
    """Cleaned liquid-hours ticks for one instrument as (ts_ns int64, logmid f64)."""
    from ..data import store
    from ..data.clean import clean_ticks
    from ..data.mids import logmid_series

    df = clean_ticks(store.read_day(instrument, date)).ticks
    ts = pd.DatetimeIndex(df["ts"])
    mask = np.asarray((ts.hour >= liquid[0]) & (ts.hour < liquid[1]))
    df = df[mask]
    p = logmid_series(df)
    return p.index.view("i8").astype(np.int64), p.to_numpy(float)

def _series_arrays(triangle: Triangle, which: str, date: dt.date, liquid):
    """(ts_ns, logmid) for a named series: 'cross' | 'leg1' | 'leg2' | 'synth'."""
    name = {"cross": triangle.cross, "leg1": triangle.leg1, "leg2": triangle.leg2}
    if which in name:
        return _instr_arrays(name[which], date, liquid)
    if which == "synth":
        t1, p1 = _instr_arrays(triangle.leg1, date, liquid)
        t2, p2 = _instr_arrays(triangle.leg2, date, liquid)
        return synth_ticks(t1, p1, t2, p2)
    raise ValueError(f"unknown series {which!r}")

def leadlag_windows(
    triangle: Triangle,
    dates: list[dt.date],
    x_series: str,
    y_series: str,
    liquid: tuple[int, int] = (7, 17),
    min_ticks: int = 200,
) -> pd.DataFrame:
    """theta* per (day x liquid window) for the pair (x_series leads if theta*<0)."""
    thetas = theta_grid_ns()
    rows, curves, skipped = [], [], 0
    for d in dates:
        try:
            tx, x = _series_arrays(triangle, x_series, d, liquid)
            ty, y = _series_arrays(triangle, y_series, d, liquid)
        except FileNotFoundError:
            skipped += 1
            continue
        if len(tx) < min_ticks or len(ty) < min_ticks:
            skipped += 1
            continue
        rho = hy_leadlag(tx, x, ty, y, thetas)
        s = leadlag_summary(thetas, rho)
        s.update(date=d, n_x=int(len(tx)), n_y=int(len(ty)))
        rows.append(s)
        curves.append(rho)
    out = pd.DataFrame(rows)
    out.attrs["thetas_ms"] = thetas / NS_PER_MS
    out.attrs["rho"] = np.array(curves) if curves else np.empty((0, len(thetas)))
    out.attrs["skipped"] = skipped
    out.attrs["pair"] = f"{x_series}~{y_series}"
    return out

def sign_test(theta_star_ms: pd.Series) -> dict:
    """Binomial sign test for 'theta* < 0 in k of n windows' (H0: p=0.5, two-sided)."""
    from scipy.stats import binomtest

    t = pd.Series(theta_star_ms).dropna()
    n = int(len(t))
    k = int((t < 0).sum())
    p = float(binomtest(k, n, 0.5).pvalue) if n else np.nan
    return {"n": n, "k_neg": k, "frac_neg": (k / n if n else np.nan), "binom_p": p}
