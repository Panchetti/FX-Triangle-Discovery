"""DST-aware trading-session labels."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SESSION_ORDER = ["TOK", "TOK+LDN", "LDN", "LDN+NY", "NY", "OFF"]

_UTC = dt.timezone.utc
_SESSIONS = {
    "TOK": ("Asia/Tokyo", (9, 0), (18, 0)),
    "LDN": ("Europe/London", (8, 0), (16, 30)),
    "NY": ("America/New_York", (8, 0), (17, 0)),
}
_ROLLOVER_TZ, _ROLLOVER_HM = "America/New_York", (17, 0)
_ROLLOVER_TOL_NS = 5 * 60 * 1_000_000_000

def _utc_ns(date: dt.date, tz: str, hm: tuple[int, int]) -> int:
    """UTC nanoseconds of a local wall-clock time on a given date (DST enters here)."""
    local = dt.datetime(date.year, date.month, date.day, hm[0], hm[1], tzinfo=ZoneInfo(tz))
    return pd.Timestamp(local.astimezone(_UTC)).value

def _bound_ns(dates: pd.Index, uniq, tz: str, hm: tuple[int, int]) -> np.ndarray:
    """Per-timestamp UTC-ns bound: compute once per unique date, map back."""
    m = {d: _utc_ns(d, tz, hm) for d in uniq}
    return dates.map(m).to_numpy(dtype="int64")

def fx_open(index: pd.DatetimeIndex) -> np.ndarray:
    """True while the FX week is open: Sun 17:00 -> Fri 17:00 America/New_York (DST-aware).

    UTC weekdays get both ends wrong (they keep frozen Friday-evening LOCF bars and drop the
    live Sunday-evening open), so trading-time masks over multi-day panels use this.
    """
    if index.tz is None:
        raise ValueError("fx_open requires a tz-aware (UTC) DatetimeIndex")
    ny = index.tz_convert(_ROLLOVER_TZ)
    wd, hr = ny.dayofweek, ny.hour
    closed = ((wd == 4) & (hr >= 17)) | (wd == 5) | ((wd == 6) & (hr < 17))
    return np.asarray(~closed)

def label_sessions(index: pd.DatetimeIndex) -> pd.Categorical:
    """One session label per timestamp; vectorized, categories = SESSION_ORDER."""
    if index.tz is None:
        raise ValueError("label_sessions requires a tz-aware (UTC) DatetimeIndex")
    idx = index.tz_convert("UTC")
    vals = idx.asi8
    dates = pd.Index(idx.date)
    uniq = dates.unique()

    def active(name: str) -> np.ndarray:
        tz, lo, hi = _SESSIONS[name]
        return (vals >= _bound_ns(dates, uniq, tz, lo)) & (vals < _bound_ns(dates, uniq, tz, hi))

    tok, ldn, ny = active("TOK"), active("LDN"), active("NY")
    roll_ctr = _bound_ns(dates, uniq, _ROLLOVER_TZ, _ROLLOVER_HM)
    roll = np.abs(vals - roll_ctr) <= _ROLLOVER_TOL_NS
    wknd = idx.dayofweek >= 5

    labels = np.select(
        [wknd | roll, ldn & ny, tok & ldn, tok, ldn, ny],
        ["OFF", "LDN+NY", "TOK+LDN", "TOK", "LDN", "NY"],
        default="OFF",
    )
    return pd.Categorical(labels, categories=SESSION_ORDER)
