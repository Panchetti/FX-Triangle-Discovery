"""DST-aware session labelling (Week 5). The DST cases ARE the point (pitfall #12)."""

from __future__ import annotations

import pandas as pd

from fxpd.calendar.sessions import SESSION_ORDER, label_sessions


def _lbl(*stamps):
    idx = pd.DatetimeIndex([pd.Timestamp(s, tz="UTC") for s in stamps])
    return list(label_sessions(idx))


def test_london_dst_shift():
    # 07:30 UTC is inside London hours in BST (opens 07:00 UTC) but not in GMT (opens 08:00 UTC)
    assert _lbl("2026-07-15 07:30") == ["TOK+LDN"]     # summer: TOK still on + LDN open
    assert _lbl("2026-01-15 07:30") == ["TOK"]          # winter: LDN not open yet


def test_us_only_dst_gap_march():
    # mid-March: US on EDT, EU still GMT -> LDN+NY still forms, shifted
    assert _lbl("2026-03-16 13:00") == ["LDN+NY"]


def test_tokyo_has_no_dst():
    # Tokyo is UTC+9 year-round -> identical label in Jan and Jul
    assert _lbl("2026-07-15 03:00") == _lbl("2026-01-15 03:00") == ["TOK"]


def test_rollover_excluded_whichever_utc_hour():
    # 5pm NY rollover: 21:00 UTC in summer (EDT), 22:00 UTC in winter (EST)
    assert _lbl("2026-07-15 21:00") == ["OFF"]          # summer rollover
    assert _lbl("2026-01-15 22:00") == ["OFF"]          # winter rollover
    assert _lbl("2026-01-15 21:00") == ["NY"]           # 21:00 UTC in winter is NOT rollover yet


def test_weekend_is_off():
    assert _lbl("2026-06-20 13:00") == ["OFF"]          # Saturday


def test_categories_and_vectorization():
    idx = pd.date_range("2026-06-15", "2026-06-16", freq="1h", tz="UTC")
    cat = label_sessions(idx)
    assert list(cat.categories) == SESSION_ORDER
    assert len(cat) == len(idx)
    # a normal weekday spans several distinct sessions
    assert {"TOK", "LDN", "NY"} <= set(cat[:-1])


def test_requires_tzaware():
    import pytest

    with pytest.raises(ValueError):
        label_sessions(pd.date_range("2026-06-15", periods=3, freq="1h"))


# --- Week 7: scheduled-announcement windows (DST-aware, date-independent) ---------
def test_news_mask_dst_shifts_us_band():
    """US 08:30 New York sits at 12:30 UTC in summer, 13:30 UTC in winter."""
    from fxpd.calendar.news import news_mask

    jul = pd.DatetimeIndex([pd.Timestamp("2026-07-15 12:30", tz="UTC")])
    jan = pd.DatetimeIndex([pd.Timestamp("2026-01-15 13:30", tz="UTC")])
    assert news_mask(jul)[0] and news_mask(jan)[0]
    # 15:00 UTC is clear of every band in BOTH seasons (nearest are ECB/US ~13:xx
    # and FOMC 18:00/19:00) -- the negative control.
    assert not news_mask(pd.DatetimeIndex([pd.Timestamp("2026-07-15 15:00", tz="UTC")]))[0]
    assert not news_mask(pd.DatetimeIndex([pd.Timestamp("2026-01-15 15:00", tz="UTC")]))[0]


def test_news_mask_pad_edges():
    from fxpd.calendar.news import news_mask

    inside = pd.DatetimeIndex([pd.Timestamp("2026-07-15 12:59", tz="UTC")])   # +29 min
    outside = pd.DatetimeIndex([pd.Timestamp("2026-07-15 13:31", tz="UTC")])  # +61 min
    assert news_mask(inside)[0] and not news_mask(outside)[0]


def test_news_mask_requires_tzaware():
    import pytest

    from fxpd.calendar.news import news_mask

    with pytest.raises(ValueError):
        news_mask(pd.date_range("2026-06-15", periods=3, freq="1h"))
