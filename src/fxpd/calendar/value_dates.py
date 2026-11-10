"""Spot (T+2) value dates, and the carry they put into the triangle basis.

The identity p_cross = p_leg1 + p_leg2 holds for rates with the SAME value date. Spot
settles two good business days after the trade date on each pair's own calendar, so when a
holiday falls inside the settlement window of only some legs, the legs settle on different
days and the quoted basis z picks up forward points for the mismatch. For a triangle
A/C = A/B * B/C (cross, leg1, leg2), with V0 = leg1's value date and n_x = V(x) - V0 in
calendar days:

    z_offset ~= [(r_C - r_A) * n_cross + (r_B - r_C) * n_leg2] / 360

(majors: A=EUR, B=USD, C=JPY). The offset is constant within a trading day, so a
per-window VECM intercept absorbs it, but it shifts the day-level basis by up to several bp
around holidays -- which a "tiny, arbitrage-bounded basis" story must account for.

Spot convention (market standard): T+1 must be a business day for the pair's non-USD
currencies -- a USD holiday on T+1 is disregarded -- and the spot date itself must be a
business day for both currencies AND for USD (non-USD crosses also settle via USD).

Holiday lists cover only the sample window (COVERAGE); dates outside it raise.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import Triangle

COVERAGE = (dt.date(2025, 12, 1), dt.date(2026, 7, 31))

# Settlement holidays (weekends handled separately). EUR = TARGET2; USD = Fedwire (open on
# the Friday before a Saturday holiday, so 3 Jul 2026 is NOT closed); JPY = Japanese bank
# holidays (incl. year-end closure and Golden Week); GBP = England & Wales bank holidays.
HOLIDAYS: dict[str, list[str]] = {
    "EUR": ["2025-12-25", "2025-12-26", "2026-01-01", "2026-04-03", "2026-04-06",
            "2026-05-01"],
    "USD": ["2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25",
            "2026-06-19"],
    "JPY": ["2025-12-31", "2026-01-01", "2026-01-02", "2026-01-12", "2026-02-11",
            "2026-02-23", "2026-03-20", "2026-04-29", "2026-05-04", "2026-05-05",
            "2026-05-06", "2026-07-20"],
    "GBP": ["2025-12-25", "2025-12-26", "2026-01-01", "2026-04-03", "2026-04-06",
            "2026-05-04", "2026-05-25"],
}


def _holidays(ccys: set[str]) -> list[str]:
    return sorted({h for c in ccys for h in HOLIDAYS[c]})


def spot_date(trade_date: dt.date, pair: str) -> dt.date:
    """Spot value date of ``pair`` (e.g. "EURUSD") traded on ``trade_date``.

    T+1 = first weekday after the trade date that is a business day for the pair's non-USD
    currencies (a USD holiday on T+1 is disregarded); spot = first weekday after T+1 that is
    a business day for both currencies and for USD. The trade date itself may be a holiday.
    """
    if not COVERAGE[0] <= trade_date <= COVERAGE[1]:
        raise ValueError(f"{trade_date} outside holiday coverage {COVERAGE}")
    ccys = {pair[:3], pair[3:]}
    unknown = ccys - set(HOLIDAYS)
    if unknown:
        raise ValueError(f"no holiday calendar for {sorted(unknown)}")
    d0 = np.datetime64(trade_date, "D")
    t1 = np.busday_offset(d0 + 1, 0, roll="forward", holidays=_holidays(ccys - {"USD"}))
    spot = np.busday_offset(t1 + 1, 0, roll="forward", holidays=_holidays(ccys | {"USD"}))
    return spot.astype(dt.date)


def value_date_mismatch(triangle: Triangle, dates: list[dt.date]) -> pd.DataFrame:
    """Per trade date: calendar-day offsets of the cross / leg2 value dates vs leg1's."""
    rows = []
    for d in dates:
        v0 = spot_date(d, triangle.leg1)
        rows.append({"date": d,
                     "n_cross": (spot_date(d, triangle.cross) - v0).days,
                     "n_leg2": (spot_date(d, triangle.leg2) - v0).days})
    return pd.DataFrame(rows).set_index("date")


def fit_carry(day_mean_bp: pd.Series, mismatch: pd.DataFrame) -> dict:
    """LS of the day-mean basis (bp) on the value-date mismatch (days), no intercept.

    No intercept because the carry model has none: zero mismatch means zero carry. As a
    check on that restriction, ``baseline_bp`` is the intercept of the same regression WITH a
    constant (should be ~0). ``r2`` is CENTERED (share of the day-to-day variance of the
    basis level explained) -- stricter than the uncentered R^2 usually quoted for
    no-intercept fits, and it can be negative if the model is useless.

    Returns per-day carry coefficients (bp/day), implied annual rate differentials in %
    (coefficient * 360 / 100), ``r2``, ``baseline_bp`` and the fitted offsets.
    For majors: diff_cross = r_JPY - r_EUR, diff_leg2 = r_USD - r_JPY.
    """
    X = mismatch.loc[day_mean_bp.index, ["n_cross", "n_leg2"]].to_numpy(float)
    y = day_mean_bp.to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    coef_c, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)
    return {
        "bp_per_day_cross": float(coef[0]), "bp_per_day_leg2": float(coef[1]),
        "diff_cross_pct": float(coef[0] * 3.6), "diff_leg2_pct": float(coef[1] * 3.6),
        "r2": r2, "baseline_bp": float(coef_c[0]),
        "pred_bp": pd.Series(pred, index=day_mean_bp.index),
    }
