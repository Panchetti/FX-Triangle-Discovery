"""Multivariate refresh-time sampling (staleness control, audit #4/#2)."""

from __future__ import annotations

import numpy as np

from fxpd.data.refresh import refresh_indices


def test_refresh_indices_known_example():
    """Hand-computed three-series case pins the algorithm and the sign of 'fresh'.

    A ticks at [1,2,5], B at [1,3,4], C at [2,3,6].
      tau_0 = max(1,1,2) = 2 ; sampled last-<=2: A=idx1, B=idx0, C=idx0
      tau_1 = max(A@5, B@3, C@3) = 5 ; sampled last-<=5: A=idx2, B=idx2, C=idx1
    Next step exhausts A -> stop. Refresh times = [2, 5].
    """
    a = np.array([1, 2, 5], np.int64)
    b = np.array([1, 3, 4], np.int64)
    c = np.array([2, 3, 6], np.int64)
    rt, idx = refresh_indices([a, b, c])
    assert rt.tolist() == [2, 5]
    assert idx[0].tolist() == [1, 2]
    assert idx[1].tolist() == [0, 2]
    assert idx[2].tolist() == [0, 1]


def test_refresh_prices_are_fresh():
    """Every sampled tick is fresh-since-prev; exactly the property that does NOT imply
    zero age at the refresh time -- so we also check the residual-age structure the
    audit-3 #1 correction is about: ages are >=0, at least one leg is age-0 per refresh."""
    rng = np.random.default_rng(0)
    times = [np.sort(rng.choice(10_000, size=800, replace=False)).astype(np.int64)
             for _ in range(3)]
    rt, idx = refresh_indices(times)
    prev = -1
    for k in range(len(rt)):
        ages = []
        for j in range(3):
            assert times[j][idx[j][k]] <= rt[k]        # not in the future
            assert times[j][idx[j][k]] > prev          # fresh since previous refresh time
            ages.append(rt[k] - times[j][idx[j][k]])
        assert min(ages) == 0                          # the leg defining the refresh is age-0
        assert max(ages) >= 0                           # others may be stale-within-cycle (>0)
        prev = rt[k]


def test_refresh_empty_series():
    rt, idx = refresh_indices([np.array([], np.int64), np.array([1, 2], np.int64)])
    assert rt.size == 0 and all(a.size == 0 for a in idx)
