"""Processed-panel cache: build-on-miss, read-on-hit, rebuild overwrites."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from fxpd.config import TRIANGLES
from fxpd.data import store, processed


def _seed_day(tmp_raw, date):
    """Write a small, well-behaved tick day for all three majors legs."""
    ts = pd.date_range(f"{date} 07:00:00", f"{date} 17:00:00", freq="30s", tz="UTC")
    rng = np.random.default_rng(0)
    for sym, base in [("EURJPY", 160.0), ("EURUSD", 1.10), ("USDJPY", 145.0)]:
        mid = base + np.cumsum(rng.normal(0, base * 1e-6, len(ts)))
        store.write_day(sym, date, pd.DataFrame({
            "ts": ts, "bid": mid - base * 1e-5, "ask": mid + base * 1e-5,
            "bid_volume": 1.0, "ask_volume": 1.0,
        }))


def test_get_panel_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(processed, "PROCESSED_DIR", tmp_path / "processed")

    tri = TRIANGLES["majors"]
    date = dt.date(2026, 6, 15)
    _seed_day(tmp_path / "raw", date)

    path = processed.panel_path(tri.name, "1min", date)
    assert not path.exists()

    p1 = processed.get_panel(tri, date, "1min")          # miss -> build + write
    assert path.exists() and not p1.empty
    assert "z" in p1.columns and "age_EURJPY" in p1.columns

    mtime = path.stat().st_mtime_ns
    p2 = processed.get_panel(tri, date, "1min")          # hit -> read, no rebuild
    assert path.stat().st_mtime_ns == mtime
    # values must round-trip; the DatetimeIndex.freq attr is not stored by parquet
    pd.testing.assert_frame_equal(p1, p2, check_freq=False)

    processed.get_panel(tri, date, "1min", rebuild=True)  # rebuild overwrites
    assert path.stat().st_mtime_ns != mtime
