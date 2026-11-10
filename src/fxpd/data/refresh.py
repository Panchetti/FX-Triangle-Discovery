"""Multivariate refresh-time sampling (Barndorff-Nielsen-Hansen-Lunde-Shephard 2011)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import Triangle

def refresh_indices(times: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Refresh times and the sampled tick index per series."""
    K = len(times)
    n = [len(a) for a in times]
    if any(v == 0 for v in n):
        return np.empty(0, np.int64), [np.empty(0, np.intp) for _ in range(K)]
    ptr = [0] * K
    rt: list[int] = []
    idx: list[list[int]] = [[] for _ in range(K)]
    while all(ptr[j] < n[j] for j in range(K)):
        tau = max(times[j][ptr[j]] for j in range(K))
        rt.append(int(tau))
        for j in range(K):
            while ptr[j] < n[j] and times[j][ptr[j]] <= tau:
                ptr[j] += 1
            idx[j].append(ptr[j] - 1)
    return np.asarray(rt, np.int64), [np.asarray(a, np.intp) for a in idx]

def refresh_panel(
    triangle: Triangle, date: dt.date, liquid: tuple[int, int] = (7, 17)
) -> pd.DataFrame:
    """Refresh-time-sampled log-mid panel for the triangle on one liquid-hours day."""
    from . import store
    from .clean import clean_ticks
    from .mids import logmid_series

    legs = list(triangle.legs)
    times, prices = [], []
    for sym in legs:
        p = logmid_series(clean_ticks(store.read_day(sym, date)).ticks)
        h = pd.DatetimeIndex(p.index).hour
        mask = np.asarray((h >= liquid[0]) & (h < liquid[1]))
        times.append(p.index.view("i8").astype(np.int64)[mask])
        prices.append(p.to_numpy(float)[mask])
    rt, idx = refresh_indices(times)
    if rt.size == 0:
        raise FileNotFoundError(f"no refresh times for {triangle.name} on {date}")
    cols = {legs[j]: prices[j][idx[j]] for j in range(len(legs))}
    ages = {f"age_{legs[j]}": (rt - times[j][idx[j]]) / 1e9 for j in range(len(legs))}
    panel = pd.DataFrame({**cols, **ages}, index=pd.DatetimeIndex(rt).tz_localize("UTC"))
    panel.index.name = "ts"
    panel["z"] = panel[triangle.cross] - panel[triangle.leg1] - panel[triangle.leg2]
    return panel

def mean_refresh_seconds(index: pd.DatetimeIndex) -> float:
    """Mean inter-refresh interval in seconds -- the 'bar_seconds' for half-life conversion."""
    if len(index) < 2:
        return float("nan")
    return float(np.diff(index.view("i8")).mean() / 1e9)
