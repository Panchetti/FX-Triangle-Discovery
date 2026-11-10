"""Triangle panel + arbitrage basis."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..config import Triangle
from .synchronize import synchronized_logmids

def build_panel(
    triangle: Triangle,
    start: dt.date,
    end: dt.date,
    freq: str,
    dropna: bool = True,
    clean: bool = True,
    with_age: bool = False,
) -> pd.DataFrame:
    """Synchronized log-mid panel for all three legs plus the basis column `z`."""
    panel = synchronized_logmids(
        list(triangle.legs), start, end, freq,
        dropna=dropna, clean=clean, with_age=with_age,
    )
    panel["z"] = panel[triangle.cross] - panel[triangle.leg1] - panel[triangle.leg2]
    return panel
