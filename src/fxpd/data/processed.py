"""Processed-panel disk cache (grid panels and refresh-time panels, one file per day)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import PROCESSED_DIR, Triangle
from .basis import build_panel
from .clean import CLEAN_VERSION

def panel_dir(triangle_name: str, freq: str) -> Path:
    return PROCESSED_DIR / f"{triangle_name}_{freq}_{CLEAN_VERSION}"

def panel_path(triangle_name: str, freq: str, date: dt.date) -> Path:
    return panel_dir(triangle_name, freq) / f"{date:%Y-%m-%d}.parquet"

def _age_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("age_")]

def get_panel(
    triangle: Triangle,
    date: dt.date,
    freq: str,
    with_age: bool = True,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Cached per-day processed panel; builds + persists on a miss.

    The cache always stores the quote-age columns (every grid driver's activity floor needs
    them), so the file is the same whatever ``with_age`` the first caller asked for; a legacy
    file written without ages is rebuilt. ``with_age=False`` only drops them on return.
    """
    path = panel_path(triangle.name, freq, date)
    panel = pd.read_parquet(path) if path.exists() and not rebuild else None
    if panel is None or not _age_cols(panel):
        panel = build_panel(triangle, date, date, freq, clean=True, with_age=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path)
    return panel if with_age else panel.drop(columns=_age_cols(panel))

def get_refresh_panel(
    triangle: Triangle,
    date: dt.date,
    liquid: tuple[int, int] = (7, 17),
    rebuild: bool = False,
) -> pd.DataFrame:
    """Cached per-day refresh-time panel (``refresh.refresh_panel``) for one liquid window."""
    from .refresh import refresh_panel

    tag = f"refresh{liquid[0]:02d}-{liquid[1]:02d}"
    path = panel_path(triangle.name, tag, date)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    panel = refresh_panel(triangle, date, liquid=liquid)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    return panel
