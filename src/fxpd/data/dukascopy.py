"""Dukascopy tick puller (resumable, cached, one provider = one clock)."""

from __future__ import annotations

import datetime as dt

import dukascopy_python as duka
import pandas as pd
from dukascopy_python.instruments import (
    INSTRUMENT_FX_CROSSES_EUR_GBP,
    INSTRUMENT_FX_CROSSES_EUR_JPY,
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
)

from . import store

DUKA_SYMBOL: dict[str, str] = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "EURJPY": INSTRUMENT_FX_CROSSES_EUR_JPY,
    "EURGBP": INSTRUMENT_FX_CROSSES_EUR_GBP,
    "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
}

_RENAME = {
    "bidPrice": "bid",
    "askPrice": "ask",
    "bidVolume": "bid_volume",
    "askVolume": "ask_volume",
}

def _daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)

def fetch_day(instrument: str, date: dt.date) -> pd.DataFrame:
    """Fetch one UTC day of ticks, normalized to the canonical store schema."""
    if instrument not in DUKA_SYMBOL:
        raise KeyError(f"Unknown instrument {instrument!r}; known: {list(DUKA_SYMBOL)}")
    start = dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    raw = duka.fetch(
        DUKA_SYMBOL[instrument], duka.INTERVAL_TICK, duka.OFFER_SIDE_BID, start, end
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=store.COLUMNS)
    df = raw.rename(columns=_RENAME).reset_index()
    df = df.rename(columns={df.columns[0]: "ts"})
    df["ts"] = pd.DatetimeIndex(df["ts"]).tz_convert("UTC")
    return df.loc[:, store.COLUMNS]

def pull_day(instrument: str, date: dt.date, force: bool = False) -> int:
    """Fetch and persist one instrument-day. Returns the number of ticks stored."""
    if not force and store.has_day(instrument, date):
        return -1
    df = fetch_day(instrument, date)
    if df.empty:
        return 0
    store.write_day(instrument, date, df)
    return len(df)

def pull(
    instruments: list[str],
    start: dt.date,
    end: dt.date,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pull a date range for several instruments. Returns a per-day tick-count log."""
    rows = []
    for instrument in instruments:
        for date in _daterange(start, end):
            n = pull_day(instrument, date, force=force)
            status = "cached" if n < 0 else ("empty" if n == 0 else "pulled")
            rows.append({"instrument": instrument, "date": date, "ticks": max(n, 0),
                         "status": status})
            if verbose:
                print(f"  {instrument} {date}  {status:>6}  {max(n, 0):>8} ticks")
    return pd.DataFrame(rows)
