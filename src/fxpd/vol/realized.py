"""Realized-kernel volatility proxy."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import Triangle

C_STAR = 3.5134  # BNHLS practical-rule constant

def parzen(x: np.ndarray) -> np.ndarray:
    """Parzen kernel weights; 1 at 0, 0 at >=1, continuous at 1/2."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    lo = (x >= 0) & (x <= 0.5)
    hi = (x > 0.5) & (x <= 1)
    out[lo] = 1 - 6 * x[lo] ** 2 + 6 * x[lo] ** 3
    out[hi] = 2 * (1 - x[hi]) ** 3
    return out

def noise_variance(r: np.ndarray) -> float:
    """omega^2 = max(-gamma_1 / n, 0): the BNHLS negative-autocovariance estimator."""
    r = np.asarray(r, float)
    n = r.size
    if n < 2:
        return 0.0
    return max(-float(np.sum(r[1:] * r[:-1])) / n, 0.0)

def bandwidth_bnhls(r: np.ndarray, iq: float) -> int:
    """H* = c* xi^(4/5) n^(3/5), xi^2 = omega^2 / sqrt(T*IQ) (T normalised to 1)."""
    r = np.asarray(r, float)
    n = r.size
    om2 = noise_variance(r)
    xi2 = om2 / np.sqrt(max(iq, 1e-300))
    h = C_STAR * (max(xi2, 0.0) ** 0.5) ** 0.8 * n ** 0.6
    return int(np.clip(round(h), 1, max(n // 3, 1)))

def realized_kernel(r: np.ndarray, H: int) -> float:
    """Parzen realized kernel of a return series at bandwidth H."""
    r = np.asarray(r, float)
    total = float(np.sum(r * r))
    for h in range(1, int(H) + 1):
        total += 2.0 * float(parzen((h - 1) / H)) * float(np.sum(r[h:] * r[:-h]))
    return total

def _sparse_rv(prices: np.ndarray, step: int) -> float:
    """RV on a sparse (subsampled) grid: sum of squared step-spaced returns."""
    sub = prices[::step]
    if sub.size < 2:
        return float("nan")
    d = np.diff(sub)
    return float(np.sum(d * d))

def rk_blocks(
    triangle: Triangle,
    dates: list[dt.date],
    freq: str = "1s",
    block: str = "30min",
    liquid: tuple[int, int] = (7, 17),
    leg: str = "cross",
    min_returns: int = 600,
    H_override: int | None = None,
) -> pd.DataFrame:
    """Realized kernel per (date x block) on the chosen leg's log mids."""
    from ..data.processed import get_panel

    bar_s = pd.Timedelta(freq).total_seconds()
    step5 = max(int(round(300 / bar_s)), 1)
    rows = []
    for d in dates:
        try:
            panel = get_panel(triangle, d, freq)
        except FileNotFoundError:
            continue
        h = panel.index.hour
        win = panel[(h >= liquid[0]) & (h < liquid[1])]
        if win.empty:
            continue
        series = win["z"] if leg == "z" else win[triangle.cross]
        for start, blk in series.groupby(pd.Grouper(freq=block)):
            p = blk.to_numpy(float)
            if p.size - 1 < min_returns:
                continue
            r = np.diff(p)
            rv5 = _sparse_rv(p, step5)
            H = (int(H_override) if H_override is not None
                 else bandwidth_bnhls(r, iq=max(rv5, 0.0) ** 2))
            rows.append({
                "date": d, "block": start, "rk": realized_kernel(r, H),
                "rv_tick": float(np.sum(r * r)), "rv_5min": rv5,
                "n": int(r.size), "H": H, "omega2": noise_variance(r),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Parzen RK only asymptotically positive; drop nonpositive finite-sample blocks
        n_bad = int((df["rk"] <= 0).sum())
        df = df[df["rk"] > 0].reset_index(drop=True)
        df.attrs["dropped_nonpositive_rk"] = n_bad
    return df

def rk_diagnostics(blocks: pd.DataFrame) -> dict:
    """Is the kernel correction actually doing anything? (see module caveat)."""
    ratio = blocks["rk"] / blocks["rv_tick"].replace(0, np.nan)
    return {
        "n_blocks": int(len(blocks)),
        "omega2_zero_share": float((blocks["omega2"] <= 0).mean()),
        "H_median": float(blocks["H"].median()),
        "rk_over_rv_median": float(ratio.median()),
    }

def sanity_gate(blocks: pd.DataFrame, lo: float = 0.5, hi: float = 2.0) -> dict:
    """RK vs subsampled 5-min RV, at BLOCK and DAY level."""
    b = blocks.dropna(subset=["rk", "rv_5min"])
    br = b["rk"] / b["rv_5min"].replace(0, np.nan)
    daily = b.groupby("date")[["rk", "rv_5min"]].sum()
    dr = daily["rk"] / daily["rv_5min"].replace(0, np.nan)
    return {
        "block_share_in_band": float(((br >= lo) & (br <= hi)).mean()),
        "day_share_in_band": float(((dr >= lo) & (dr <= hi)).mean()),
        "n_days": int(len(daily)),
    }
