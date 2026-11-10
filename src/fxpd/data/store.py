"""Partitioned tick store."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR

COLUMNS = ["ts", "bid", "ask", "bid_volume", "ask_volume"]

def day_path(instrument: str, date: dt.date) -> Path:
    """Parquet path for one instrument-day."""
    return RAW_DIR / instrument / f"{date:%Y}" / f"{instrument}-{date:%Y-%m-%d}.parquet"

def has_day(instrument: str, date: dt.date) -> bool:
    return day_path(instrument, date).exists()

def write_day(instrument: str, date: dt.date, df: pd.DataFrame) -> Path:
    """Write one instrument-day. `df` must carry the canonical COLUMNS."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{instrument} {date}: missing columns {missing}")
    df = df.loc[:, COLUMNS].sort_values("ts").reset_index(drop=True)
    path = day_path(instrument, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path

def read_day(instrument: str, date: dt.date) -> pd.DataFrame:
    path = day_path(instrument, date)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)

def available_days(instrument: str) -> list[dt.date]:
    """All stored dates for an instrument, ascending."""
    root = RAW_DIR / instrument
    if not root.exists():
        return []
    days = []
    for p in root.rglob(f"{instrument}-*.parquet"):
        stamp = p.stem.rsplit("-", 3)[-3:]
        days.append(dt.date(int(stamp[0]), int(stamp[1]), int(stamp[2])))
    return sorted(days)

def read_range(
    instrument: str, start: dt.date, end: dt.date
) -> pd.DataFrame:
    """Concatenate all stored days in [start, end] (inclusive) for one instrument."""
    frames = [
        read_day(instrument, d)
        for d in available_days(instrument)
        if start <= d <= end
    ]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("ts").reset_index(drop=True)
