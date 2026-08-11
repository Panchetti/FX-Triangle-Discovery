"""Planted-defect tests for fxpd.data.clean + the quote-age infra (Week 2 gate).

A synthetic EURUSD-like day (~500 ticks, mid ~1.10, spread ~1e-4, 1s spacing)
gets exactly four planted defects at known rows; each must be caught by its
dedicated rule and nothing else may be dropped.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fxpd.data import store
from fxpd.data.clean import clean_ticks
from fxpd.data.synchronize import synchronized_logmids

N = 500
I_NAN, I_CROSSED, I_SPREAD, I_SPIKE = 120, 150, 300, 400


def _base_day(n: int = N) -> pd.DataFrame:
    """Well-behaved ticks: tiny random-walk mid, constant 1e-4 spread, 1s apart."""
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-06-15 07:00:00", periods=n, freq="1s", tz="UTC")
    mid = 1.10 + np.cumsum(rng.normal(0, 2e-6, n))
    return pd.DataFrame(
        {
            "ts": ts,
            "bid": mid - 0.5e-4,
            "ask": mid + 0.5e-4,
            "bid_volume": np.ones(n),
            "ask_volume": np.ones(n),
        }
    )


def _defective_day() -> pd.DataFrame:
    df = _base_day()
    df.loc[I_NAN, "bid"] = np.nan                                   # rule 1
    df.loc[I_CROSSED, ["bid", "ask"]] = [1.1001, 1.1000]            # rule 2
    mid = 0.5 * (df.loc[I_SPREAD, "bid"] + df.loc[I_SPREAD, "ask"])
    df.loc[I_SPREAD, ["bid", "ask"]] = [mid - 25e-4, mid + 25e-4]   # rule 3: 50x spread
    df.loc[I_SPIKE, ["bid", "ask"]] += 30e-4 * 1.10                 # rule 4: 30 bp spike
    return df


def test_planted_defects_each_caught_by_its_rule():
    df = _defective_day()
    res = clean_ticks(df)
    assert res.counts == {
        "input": N, "validity": 1, "crossed": 1, "spread": 1, "spike": 1,
        "output": N - 4,
    }
    dropped_ts = set(df["ts"]) - set(res.ticks["ts"])
    assert dropped_ts == set(df.loc[[I_NAN, I_CROSSED, I_SPREAD, I_SPIKE], "ts"])
    assert len(res.ticks) == N - 4


def test_clean_is_noop_on_defect_free_data():
    df = _base_day()
    res = clean_ticks(df)
    assert res.counts["output"] == N
    assert sum(res.counts[r] for r in ("validity", "crossed", "spread", "spike")) == 0
    pd.testing.assert_frame_equal(res.ticks, df)


def test_spike_filter_toggle_keeps_the_spike(monkeypatch=None):
    """spike_filter=False disables rule 4 so the planted spike survives (audit-4 #14).

    Enables the no-spike-filter sensitivity: the same defect frame keeps the 30 bp spike
    (validity/crossed/spread still drop their 3), so exactly one more row survives.
    """
    df = _defective_day()
    on = clean_ticks(df)
    off = clean_ticks(df, spike_filter=False)
    assert off.counts["spike"] == 0
    assert off.counts["output"] == on.counts["output"] + 1     # the spike is retained
    surviving = set(off.ticks["ts"])
    assert df.loc[I_SPIKE, "ts"] in surviving                  # the spike row is back


def test_quote_age_hand_computed(tmp_path, monkeypatch):
    import fxpd.data.store as s

    monkeypatch.setattr(s, "RAW_DIR", tmp_path)
    date = dt.date(2026, 6, 15)
    ts = pd.DatetimeIndex(
        ["2026-06-15T00:00:00.4", "2026-06-15T00:00:01.7", "2026-06-15T00:00:03.1"],
        tz="UTC",
    )
    store.write_day(
        "EURUSD",
        date,
        pd.DataFrame(
            {"ts": ts, "bid": [1.1, 1.2, 1.3], "ask": [1.11, 1.21, 1.31],
             "bid_volume": 1.0, "ask_volume": 1.0}
        ),
    )
    panel = synchronized_logmids(
        ["EURUSD"], date, date, "1s", dropna=False, clean=False, with_age=True
    )
    age = panel["age_EURUSD"]
    assert np.isnan(age.iloc[0])  # grid 00:00:00 precedes the first tick
    assert age.iloc[1] == pytest.approx(0.6)  # 01.0 - 00.4
    assert age.iloc[2] == pytest.approx(0.3)  # 02.0 - 01.7
    assert age.iloc[3] == pytest.approx(1.3)  # 03.0 - 01.7
    assert age.iloc[4] == pytest.approx(0.9)  # 04.0 - 03.1
