"""Tick cleaning and per-day QC reports."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import QC_DIR
from . import store

CLEAN_VERSION = "v1"

_BP = 1e-4

@dataclass
class CleanResult:
    """Cleaned ticks (canonical store columns, ts-sorted) + per-rule drop counts."""

    ticks: pd.DataFrame
    counts: dict[str, int]

def clean_ticks(
    df: pd.DataFrame,
    *,
    spread_k: float = 10.0,
    spread_window: str = "1h",
    spread_min_periods: int = 100,
    spike_neighbors: int = 20,
    spike_c: float = 10.0,
    spike_g_bp: float = 0.05,
    spike_filter: bool = True,
) -> CleanResult:
    """Apply the filters in order; count every drop. See module docstring."""
    counts: dict[str, int] = {"input": int(len(df))}
    if df.empty:
        counts.update(validity=0, crossed=0, spread=0, spike=0, output=0)
        return CleanResult(df.loc[:, store.COLUMNS].copy(), counts)

    t = df.loc[:, store.COLUMNS].sort_values("ts").reset_index(drop=True)

    ok = t["bid"].notna() & t["ask"].notna() & (t["bid"] > 0) & (t["ask"] > 0)
    counts["validity"] = int((~ok).sum())
    t = t[ok]

    ok = t["ask"] > t["bid"]
    counts["crossed"] = int((~ok).sum())
    t = t[ok]

    ti = t.set_index("ts")
    spread = ti["ask"] - ti["bid"]
    med = spread.rolling(spread_window, min_periods=spread_min_periods).median()
    ok = ~(spread > spread_k * med)
    counts["spread"] = int((~ok).sum())
    ti = ti[ok.to_numpy()]

    if spike_filter:
        mid = 0.5 * (ti["bid"] + ti["ask"])
        win = 2 * spike_neighbors + 1
        med = mid.rolling(win, center=True, min_periods=spike_neighbors).median()
        mad = (mid - med).abs().rolling(win, center=True, min_periods=spike_neighbors).median()
        g = spike_g_bp * _BP * med
        ok = ~((mid - med).abs() > spike_c * (mad + g))
        counts["spike"] = int((~ok).sum())
        ti = ti[ok.to_numpy()]
    else:
        counts["spike"] = 0

    out = ti.reset_index().loc[:, store.COLUMNS]
    counts["output"] = int(len(out))
    return CleanResult(out, counts)

def qc_report(instrument: str, date: dt.date, result: CleanResult) -> dict:
    """QC dict for one (instrument, day): counts + spread/gap/coverage stats."""
    t = result.ticks
    rep: dict = {
        "instrument": instrument,
        "date": date.isoformat(),
        "clean_version": CLEAN_VERSION,
        **{f"n_{k}": v for k, v in result.counts.items()},
    }
    if t.empty:
        rep.update(
            max_gap_s=np.nan, spread_p50_bp=np.nan, spread_p95_bp=np.nan,
            spread_p99_bp=np.nan, first_tick=None, last_tick=None,
            crossed_share_flag=False,
        )
        return rep
    ts = pd.DatetimeIndex(t["ts"])
    spread_bp = (t["ask"] - t["bid"]) / (0.5 * (t["ask"] + t["bid"])) / _BP
    gaps = ts.to_series().diff().dt.total_seconds()
    rep.update(
        max_gap_s=float(gaps.max()) if len(t) > 1 else np.nan,
        spread_p50_bp=float(spread_bp.quantile(0.50)),
        spread_p95_bp=float(spread_bp.quantile(0.95)),
        spread_p99_bp=float(spread_bp.quantile(0.99)),
        first_tick=ts[0].isoformat(),
        last_tick=ts[-1].isoformat(),
        crossed_share_flag=bool(
            result.counts["crossed"] > 0.001 * max(result.counts["input"], 1)
        ),
    )
    return rep

def _qc_path(instrument: str, date: dt.date):
    return QC_DIR / f"{instrument}-{date.isoformat()}.json"

def run_qc(
    instruments: list[str], start: dt.date, end: dt.date, force: bool = False
) -> pd.DataFrame:
    """Clean every stored (instrument, day) in range and persist QC JSONs."""
    QC_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for instrument in instruments:
        for date in store.available_days(instrument):
            if not (start <= date <= end):
                continue
            path = _qc_path(instrument, date)
            if path.exists() and not force:
                rows.append(json.loads(path.read_text()))
                continue
            rep = qc_report(instrument, date, clean_ticks(store.read_day(instrument, date)))
            path.write_text(json.dumps(rep, indent=1))
            rows.append(rep)
    return pd.DataFrame(rows)

def qc_summary() -> pd.DataFrame:
    """All persisted QC reports, one row per (instrument, day)."""
    rows = [json.loads(p.read_text()) for p in sorted(QC_DIR.glob("*.json"))]
    return pd.DataFrame(rows)
