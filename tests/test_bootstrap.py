"""Block-bootstrap inference (Week 5). The coverage contrast IS pitfall #7 as a test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import IIDBootstrap, MovingBlockBootstrap

from fxpd.inference.bootstrap import (
    diff_medians,
    diff_vs_equal_edge,
    holm,
    metric_by_day,
    p_from_dist,
)


def test_diff_vs_equal_edge_is_complete_case_and_equal_weight():
    """Equal-weight, complete-case edge contrast (audit-4 #1).

    Only days where target AND every edge are present enter; the edge benchmark is the
    UNWEIGHTED mean of the edge sessions, so a comparator's day count cannot tilt it.
    """
    idx = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    w = pd.DataFrame({"LDNNY": [np.nan, 4.0, 6.0],   # day 0 dropped (target missing)
                      "TOK":   [1.0, 2.0, 2.0],
                      "NY":    [3.0, 4.0, 10.0]}, index=idx)
    # day1: 4 - (2+4)/2 = 1 ; day2: 6 - (2+10)/2 = 0 ; median([1,0]) = 0.5
    assert diff_vs_equal_edge("LDNNY", ["TOK", "NY"])(w) == 0.5
    # equal weight: swapping which edge is large must not change the benchmark mean
    w2 = w.rename(columns={"TOK": "NY", "NY": "TOK"})
    assert diff_vs_equal_edge("LDNNY", ["TOK", "NY"])(w2) == 0.5
    # all-missing target -> NaN, never a spurious number
    wn = w.assign(LDNNY=np.nan)
    assert np.isnan(diff_vs_equal_edge("LDNNY", ["TOK", "NY"])(wn))


def _ar1(n, phi, mean, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x + mean


def test_holm_matches_known_values():
    # classic Holm example
    adj = holm([0.01, 0.02, 0.03, 0.04])
    assert np.allclose(adj, [0.04, 0.06, 0.06, 0.06])
    # monotone, capped at 1
    assert holm([0.5, 0.6])[1] <= 1.0


def test_metric_by_day_shapes_and_nan():
    res = {
        "TOK": pd.DataFrame({"date": [1, 2, 3], "gg_cross": [0.7, 0.6, 0.65], "valid": [True, True, True]}),
        "LDN": pd.DataFrame({"date": [1, 2, 3], "gg_cross": [0.6, 0.5, 0.55], "valid": [True, False, True]}),
    }
    wide = metric_by_day(res, "gg_cross")
    assert list(wide.columns) == ["TOK", "LDN"]
    assert np.isnan(wide.loc[2, "LDN"])          # invalid window -> NaN
    assert diff_medians("TOK", "LDN")(wide) > 0  # TOK median > LDN median here


def test_p_from_dist_directions():
    d = np.array([-1.0, 0.5, 0.5, 0.5])  # mostly positive
    assert p_from_dist(d, "greater") == 0.25   # share <= 0
    assert p_from_dist(d, "less") == 0.75       # share >= 0
    assert p_from_dist(d, "two-sided") == 0.5


def test_block_covers_iid_undercovers():
    """Two AR(1) series with a known mean gap delta; the diff-of-means CI should
    cover delta ~95% under the DAY-block bootstrap, and the naive iid bootstrap
    (which ignores serial dependence) UNDER-covers. That contrast justifies blocking.
    """
    n, phi, delta, block = 400, 0.7, 1.0, 25
    R, B = 120, 199

    def _stat(a, b):
        return np.array([b.mean() - a.mean()])

    def coverage(use_block):
        hits = 0
        for r in range(R):
            a = _ar1(n, phi, 0.0, seed=2 * r)
            b = _ar1(n, phi, delta, seed=2 * r + 1)
            boot = (MovingBlockBootstrap(block, a, b, seed=r) if use_block
                    else IIDBootstrap(a, b, seed=r))
            ci = boot.conf_int(_stat, B, method="percentile")
            hits += ci[0, 0] <= delta <= ci[1, 0]
        return hits / R

    cov_block = coverage(True)
    cov_iid = coverage(False)
    # block bootstrap ~ nominal (percentile CIs under-cover a little in finite samples);
    # iid bootstrap SEVERELY under-covers because it ignores the serial dependence.
    assert cov_block > 0.85, f"block coverage too low: {cov_block}"
    assert cov_iid < 0.78, f"iid should under-cover badly: {cov_iid}"
    assert cov_block - cov_iid > 0.15, f"contrast too small: block={cov_block} iid={cov_iid}"
