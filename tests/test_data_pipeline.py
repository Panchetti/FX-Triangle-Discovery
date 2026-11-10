"""Fast, network-free tests for the data layer.

These build synthetic ticks so the alignment / basis math is checked
deterministically -- no Dukascopy calls in CI.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from fxpd.config import Triangle
from fxpd.data import store
from fxpd.data.basis import build_panel
from fxpd.data.synchronize import previous_tick, build_grid


def _ticks(ts, bids, asks):
    return pd.DataFrame(
        {
            "ts": pd.DatetimeIndex(ts, tz="UTC"),
            "bid": bids,
            "ask": asks,
            "bid_volume": np.ones(len(ts)),
            "ask_volume": np.ones(len(ts)),
        }
    )


def test_previous_tick_locf():
    ticks = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.DatetimeIndex(
            ["2026-06-15T00:00:00.4", "2026-06-15T00:00:01.7", "2026-06-15T00:00:03.1"],
            tz="UTC",
        ),
    )
    grid = pd.date_range("2026-06-15T00:00:00", periods=5, freq="1s", tz="UTC")
    out = previous_tick(ticks, grid)
    # t=0 has no prior tick -> NaN; then LOCF
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 10.0  # last tick before 00:01 is the .4 tick
    assert out.iloc[2] == 11.0  # last before 00:02 is the 1.7 tick
    assert out.iloc[3] == 11.0
    assert out.iloc[4] == 12.0


def test_basis_identity_is_zero_when_consistent(tmp_path, monkeypatch):
    """If cross == leg1 * leg2 exactly at every tick, the basis must be ~0."""
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    # also patch the module-level name used inside store.day_path
    import fxpd.data.store as s
    monkeypatch.setattr(s, "RAW_DIR", tmp_path)

    date = dt.date(2026, 6, 15)
    grid = build_grid(date, date, "1s")[:50]
    rng = np.random.default_rng(0)
    p_leg1 = np.cumsum(rng.normal(0, 1e-4, len(grid))) + np.log(1.10)  # EURUSD
    p_leg2 = np.cumsum(rng.normal(0, 1e-4, len(grid))) + np.log(160.0)  # USDJPY
    p_cross = p_leg1 + p_leg2  # EURJPY consistent by construction

    for sym, logp in [("EURUSD", p_leg1), ("USDJPY", p_leg2), ("EURJPY", p_cross)]:
        mid = np.exp(logp)
        store.write_day(sym, date, _ticks(grid, mid * (1 - 1e-5), mid * (1 + 1e-5)))

    tri = Triangle("majors", cross="EURJPY", leg1="EURUSD", leg2="USDJPY")
    panel = build_panel(tri, date, date, "1s")
    # half-spread is symmetric in price, so log-mid basis is essentially exact
    assert panel["z"].abs().max() < 1e-6
