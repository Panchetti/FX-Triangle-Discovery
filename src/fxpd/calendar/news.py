"""Scheduled-announcement windows -- DST-aware, date-independent."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

RELEASE_TIMES: list[tuple[str, str, tuple[int, int]]] = [
    ("US_macro_0830", "America/New_York", (8, 30)),
    ("FOMC_1400", "America/New_York", (14, 0)),
    ("ECB_1415", "Europe/Berlin", (14, 15)),
    ("ECB_presser_1445", "Europe/Berlin", (14, 45)),
    ("BOE_1200", "Europe/London", (12, 0)),
    ("BOJ_1130", "Asia/Tokyo", (11, 30)),
]

PAD = pd.Timedelta("30min")

def _utc_ns(date: dt.date, tz: str, hm: tuple[int, int]) -> int:
    local = dt.datetime(date.year, date.month, date.day, hm[0], hm[1], tzinfo=ZoneInfo(tz))
    return pd.Timestamp(local.astimezone(dt.timezone.utc)).value

def news_mask(index: pd.DatetimeIndex, pad: pd.Timedelta = PAD) -> np.ndarray:
    """True where a timestamp falls within ``pad`` of ANY scheduled release time."""
    if index.tz is None:
        raise ValueError("news_mask requires a tz-aware (UTC) DatetimeIndex")
    idx = index.tz_convert("UTC")
    vals = idx.asi8
    dates = pd.Index(idx.date)
    uniq = dates.unique()
    pad_ns = int(pad.value)

    out = np.zeros(vals.shape, dtype=bool)
    for _, tz, hm in RELEASE_TIMES:
        centres = dates.map({d: _utc_ns(d, tz, hm) for d in uniq}).to_numpy(dtype="int64")
        out |= np.abs(vals - centres) <= pad_ns
    return out

def news_free_transitions(dates: list[dt.date], pad: pd.Timedelta = PAD) -> pd.Series:
    """Transition-point series for ``fit_discovery_windows(bar_mask=...)``."""
    pts: dict[pd.Timestamp, bool] = {}
    for d in dates:
        for _, tz, hm in RELEASE_TIMES:
            c = pd.Timestamp(_utc_ns(d, tz, hm), tz="UTC")
            pts[c - pad] = False
            pts[c + pad] = True
    if dates:
        pts[pd.Timestamp(min(dates), tz="UTC")] = True
    return pd.Series(pts).sort_index()

def write_rule_calendar(path, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Rule-derived event list (first-Friday NFP, monthly CPI proxy, etc.)."""
    rows = []
    for ts in pd.date_range(start, end, freq="D"):
        d = ts.date()
        if ts.weekday() == 4 and ts.day <= 7:                    # first Friday
            rows.append((d, "US_NFP", "US", "America/New_York", "08:30"))
        if ts.weekday() == 2 and 8 <= ts.day <= 14:
            rows.append((d, "US_CPI_proxy", "US", "America/New_York", "08:30"))
    df = pd.DataFrame(rows, columns=["date", "event", "region", "tz", "local_time"])
    header = (
        "# Rule-derived calendar (unverified). Dates from scheduling rules,\n"
        "# NOT confirmed against a source. Primary news test is time-of-day band\n"
        "# exclusion (fxpd.calendar.news.news_mask), which needs no date list.\n"
        "# Replace with a sourced central-bank calendar before publication.\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    return df
