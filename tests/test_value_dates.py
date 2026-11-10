"""Spot value dates + carry fit: known-answer cases from the 2026 holiday calendar."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fxpd.calendar.value_dates import fit_carry, spot_date, value_date_mismatch
from fxpd.config import TRIANGLES

TRI = TRIANGLES["majors"]


def test_jpy_holiday_in_window_splits_value_dates():
    # Thu 8 Jan; Mon 12 Jan is a Japanese holiday -> JPY pairs settle a day later
    d = dt.date(2026, 1, 8)
    assert spot_date(d, "EURUSD") == dt.date(2026, 1, 12)
    assert spot_date(d, "USDJPY") == dt.date(2026, 1, 13)
    assert spot_date(d, "EURJPY") == dt.date(2026, 1, 13)


def test_trading_on_the_holiday_itself_realigns():
    # counting starts the day AFTER the trade date, so trading on 12 Jan -> all settle 14 Jan
    m = value_date_mismatch(TRI, [dt.date(2026, 1, 12)])
    assert m.iloc[0].tolist() == [0, 0]


def test_usd_holiday_shifts_every_leg_together():
    # Memorial Day (Mon 25 May) is a USD holiday: all three pairs avoid it -> no mismatch
    m = value_date_mismatch(TRI, [dt.date(2026, 5, 21)])
    assert m.iloc[0].tolist() == [0, 0]


def test_usd_holiday_on_t_plus_1_is_disregarded():
    # Fri 16 Jan; Mon 19 Jan (MLK) is a USD holiday on T+1 -> ignored -> spot Tue 20 Jan
    for pair in ("EURUSD", "USDJPY", "EURJPY"):
        assert spot_date(dt.date(2026, 1, 16), pair) == dt.date(2026, 1, 20)


def test_christmas_eve_splits_usdjpy_from_eur_legs():
    # Wed 24 Dec: T+1 = 25 Dec counts for USDJPY (JPY open; USD holiday ignored on T+1) ->
    # spot Fri 26 Dec; EUR legs skip 25/26 Dec (TARGET) -> spot Tue 30 Dec
    d = dt.date(2025, 12, 24)
    assert spot_date(d, "USDJPY") == dt.date(2025, 12, 26)
    assert spot_date(d, "EURUSD") == dt.date(2025, 12, 30)
    assert value_date_mismatch(TRI, [d]).iloc[0].tolist() == [0, -4]


def test_gbp_triangle_and_unknown_currency():
    m = value_date_mismatch(TRIANGLES["gbp"], [dt.date(2026, 4, 30)])   # UK 4 May holiday
    assert m.shape == (1, 2)
    with pytest.raises(ValueError):
        spot_date(dt.date(2026, 1, 8), "AUDUSD")


def test_outside_coverage_raises():
    with pytest.raises(ValueError):
        spot_date(dt.date(2027, 1, 4), "EURUSD")


def test_fit_carry_recovers_known_differentials():
    dates = list(pd.bdate_range("2025-12-15", "2026-07-20").date)
    m = value_date_mismatch(TRI, dates)
    true = {"cross": -0.37, "leg2": 0.90}                       # bp per day of mismatch
    y = pd.Series(true["cross"] * m["n_cross"] + true["leg2"] * m["n_leg2"], index=m.index)
    y = y + np.random.default_rng(0).normal(0, 1e-6, len(y))   # nonzero SS_tot on flat days
    fit = fit_carry(y, m)
    assert fit["bp_per_day_cross"] == pytest.approx(true["cross"], abs=1e-4)
    assert fit["bp_per_day_leg2"] == pytest.approx(true["leg2"], abs=1e-4)
    assert fit["diff_leg2_pct"] == pytest.approx(0.90 * 3.6, abs=1e-3)
    assert fit["r2"] > 0.999
    assert abs(fit["baseline_bp"]) < 1e-4                      # no constant in the true DGP
