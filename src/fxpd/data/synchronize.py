"""Previous-tick synchronization onto a common clock."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import store
from .mids import logmid_series

def previous_tick(series: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Sample `series` (tick-indexed, UTC) onto `grid` via LOCF."""
    series = series.sort_index()
    combined = series[~series.index.duplicated(keep="last")]
    reindexed = combined.reindex(combined.index.union(grid)).ffill()
    return reindexed.reindex(grid)

def build_grid(start: dt.date, end: dt.date, freq: str) -> pd.DatetimeIndex:
    """Regular UTC grid spanning [start 00:00, end 23:59:59...] at `freq`."""
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return pd.date_range(lo, hi, freq=freq, inclusive="left", tz="UTC")

def synchronized_logmids(
    instruments: list[str],
    start: dt.date,
    end: dt.date,
    freq: str,
    dropna: bool = True,
    clean: bool = True,
    with_age: bool = False,
) -> pd.DataFrame:
    """One log-mid column per instrument, previous-tick aligned on a common grid."""
    from .clean import clean_ticks

    grid = build_grid(start, end, freq)
    cols = {}
    for ins in instruments:
        ticks = store.read_range(ins, start, end)
        if ticks.empty:
            raise FileNotFoundError(f"No stored ticks for {ins} in [{start}, {end}]")
        if clean:
            ticks = clean_ticks(ticks).ticks
        cols[ins] = previous_tick(logmid_series(ticks), grid)
        if with_age:
            tick_ts = pd.DatetimeIndex(ticks["ts"])
            locf_ts = previous_tick(pd.Series(tick_ts, index=tick_ts), grid)
            cols[f"age_{ins}"] = (grid - pd.DatetimeIndex(locf_ts)).total_seconds()
    panel = pd.DataFrame(cols, index=grid)
    panel.index.name = "ts"
    return panel.dropna() if dropna else panel
