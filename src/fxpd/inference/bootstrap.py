"""Moving-block bootstrap over days, plus Holm multiplicity control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from arch.bootstrap import IIDBootstrap, MovingBlockBootstrap

@dataclass
class HypResult:
    name: str
    stat: float
    ci_lo: float
    ci_hi: float
    p_raw: float
    p_holm: float | None = None

def metric_by_day(res: dict[str, pd.DataFrame], metric: str = "gg_cross") -> pd.DataFrame:
    """Wide frame index=date, columns=sessions, values=`metric` over VALID windows."""
    cols = {}
    for session, df in res.items():
        if df.empty or "date" not in df.columns:
            cols[session] = pd.Series(dtype=float)
            continue
        valid = df[df["valid"]] if "valid" in df.columns else df
        cols[session] = valid.set_index("date")[metric]
    return pd.DataFrame(cols).sort_index()

def block_bootstrap(
    wide: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    block: int = 5,
    B: int = 2000,
    seed: int = 0,
) -> tuple[float, tuple[float, float], np.ndarray]:
    """Moving-block bootstrap of a scalar statistic over day-rows."""
    def _f(df):
        return np.array([stat_fn(df)])

    mbb = MovingBlockBootstrap(block, wide, seed=seed)
    dist = mbb.apply(_f, B).ravel()
    point = float(stat_fn(wide))
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return point, (float(lo), float(hi)), dist

def iid_bootstrap(
    wide: pd.DataFrame, stat_fn, B: int = 2000, seed: int = 0
) -> tuple[float, tuple[float, float], np.ndarray]:
    """i.i.d. bootstrap (the WRONG one for dependent days) -- for the coverage contrast."""
    def _f(df):
        return np.array([stat_fn(df)])

    bs = IIDBootstrap(wide, seed=seed)
    dist = bs.apply(_f, B).ravel()
    point = float(stat_fn(wide))
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return point, (float(lo), float(hi)), dist

def p_from_dist(dist: np.ndarray, alternative: str = "two-sided") -> float:
    """CI-inversion bootstrap p-value. alternative in {greater, less, two-sided}."""
    dist = np.asarray(dist, float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return float("nan")
    if alternative == "greater":
        return float((dist <= 0).mean())
    if alternative == "less":
        return float((dist >= 0).mean())
    return float(2 * min((dist <= 0).mean(), (dist >= 0).mean()))

def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (valid under any dependence)."""
    p = np.asarray(pvals, float)
    adj = np.full(p.size, np.nan)
    idx_ok = np.flatnonzero(np.isfinite(p))
    if idx_ok.size == 0:
        return adj.tolist()
    sub = p[idx_ok]
    m = sub.size
    order = np.argsort(sub)
    out = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * sub[j])
        out[j] = min(running, 1.0)
    adj[idx_ok] = out
    return adj.tolist()

def diff_medians(session_a: str, session_b: str) -> Callable[[pd.DataFrame], float]:
    """stat_fn: nanmedian(A) - nanmedian(B) (the pinned between-session difference)."""
    return lambda w: float(np.nanmedian(w[session_a]) - np.nanmedian(w[session_b]))

def diff_median_vs_pool(session_a: str, pool: list[str]) -> Callable[[pd.DataFrame], float]:
    """stat_fn: nanmedian(A) - nanmedian(pooled columns)."""
    return lambda w: float(np.nanmedian(w[session_a]) - np.nanmedian(w[pool].to_numpy()))

def diff_vs_equal_edge(target: str, edges: list[str]) -> Callable[[pd.DataFrame], float]:
    """stat_fn: median over complete-case days of [target - mean(edges)]."""
    cols = [target, *edges]

    def _f(w: pd.DataFrame) -> float:
        sub = w[cols].dropna()
        if sub.empty:
            return float("nan")
        edge = sub[edges].mean(axis=1)
        return float(np.median(sub[target] - edge))

    return _f

def session_median_ci(
    wide: pd.DataFrame, session: str, block: int = 5, B: int = 2000, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Block-bootstrap CI of one session's median (the money-plot error bars)."""
    point, ci, _ = block_bootstrap(
        wide, lambda w: float(np.nanmedian(w[session])), block=block, B=B, seed=seed
    )
    return point, ci
