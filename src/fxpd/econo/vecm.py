"""VECM with imposed beta: who error-corrects in the triangle."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

from ..config import Triangle

TRIVARIATE = "trivariate"  # columns (cross, leg1, leg2), beta = (1, -1, -1)
BIVARIATE = "bivariate"    # columns (cross, synth),      beta = (1, -1)

LJUNGBOX_LAGS = 20

@dataclass
class VECMWindowResult:
    """One estimation window's alpha loadings + diagnostics."""

    date: dt.date
    frame: str
    k: int
    n_obs: int
    names: list[str]
    alpha: np.ndarray
    alpha_se: np.ndarray
    alpha_t: np.ndarray
    beta_alpha: float
    half_life_s: float
    ljungbox_p: np.ndarray
    fresh_share: np.ndarray

def half_life_seconds(beta_alpha: float, bar_seconds: float) -> float:
    """HL = ln 2 / (-ln(1 + beta'alpha)) bars -> seconds; NaN outside (-1, 0)."""
    if not (-1.0 < beta_alpha < 0.0):
        return float("nan")
    return float(np.log(2) / (-np.log1p(beta_alpha)) * bar_seconds)

def dynamic_half_life(
    alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
    bar_seconds: float, max_h: int = 20000, interpolate: bool = True,
    rebound_bars: int = 200, return_diagnostics: bool = False,
):
    """First-passage half-time of z in the fitted VECM, in seconds."""
    alpha = np.asarray(alpha, float).ravel()
    beta = np.asarray(beta, float).ravel()
    k, m = gamma.shape[0], alpha.size
    dx = [np.zeros(m) for _ in range(k)]
    z = 1.0
    hl = float("nan")
    diag = {"crossed": False, "censored": True, "rebound": False,
            "bars_to_cross": None, "max_abs_z_after_cross": 0.0}
    cross_h = None
    for h in range(1, int(max_h) + 1):
        z_prev = z
        d = alpha * z + sum(gamma[i] @ dx[-1 - i] for i in range(k))
        z = z + float(beta @ d)
        dx.append(d)
        if cross_h is None and abs(z) <= 0.5:
            cross_h = h
            diag.update(crossed=True, censored=False, bars_to_cross=h)
            if interpolate:
                a, b = abs(z_prev), abs(z)
                frac = (a - 0.5) / (a - b) if a > b else 1.0
                hl = (h - 1 + min(max(frac, 0.0), 1.0)) * bar_seconds
            else:
                hl = h * bar_seconds
            if not return_diagnostics:
                return float(hl)
        if cross_h is not None:
            diag["max_abs_z_after_cross"] = max(diag["max_abs_z_after_cross"], abs(z))
            if abs(z) > 0.5:
                diag["rebound"] = True
            if h >= cross_h + rebound_bars:
                break
    if return_diagnostics:
        return float(hl), diag
    return float(hl)

def make_frame(panel: pd.DataFrame, triangle: Triangle, frame: str) -> pd.DataFrame:
    """Tidy estimation frame: price columns (canonical order) + "z"."""
    c, l1, l2 = triangle.cross, triangle.leg1, triangle.leg2
    if frame == TRIVARIATE:
        out = panel[[c, l1, l2]].copy()
        out["z"] = out[c] - out[l1] - out[l2]
    elif frame == BIVARIATE:
        out = pd.DataFrame(index=panel.index)
        out[c] = panel[c]
        out["synth"] = panel[l1] + panel[l2]
        out["z"] = out[c] - out["synth"]
    else:
        raise ValueError(f"frame must be {TRIVARIATE!r} or {BIVARIATE!r}, got {frame!r}")
    return out

def frame_names(triangle: Triangle, frame: str) -> list[str]:
    """Equation names in canonical order for a frame."""
    if frame == TRIVARIATE:
        return list(triangle.legs)
    return [triangle.cross, "synth"]

def _design(P: np.ndarray, z: np.ndarray, k: int, t0: int):
    """Shared regression arrays for rows t = t0+1 .. T-1 (bar indices)."""
    T = len(z)
    dP = np.diff(P, axis=0)
    n = T - 1 - t0
    y = dP[t0:]
    cols = [np.ones((n, 1)), z[t0 : T - 1, None]]
    cols += [dP[t0 - i : T - 1 - i] for i in range(1, k + 1)]
    return y, np.hstack(cols)

def select_k_bic(x: pd.DataFrame, k_max: int = 20) -> pd.DataFrame:
    """Multivariate BIC over k in 1..k_max on ONE window."""
    price_cols = [c for c in x.columns if c != "z"]
    P = x[price_cols].to_numpy(float)
    z = x["z"].to_numpy(float)
    m = P.shape[1]
    y, X_full = _design(P, z, k_max, k_max)
    n = len(y)
    rows = []
    for k in range(1, k_max + 1):
        X = X_full[:, : 2 + m * k]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        E = y - X @ coef
        sign, logdet = np.linalg.slogdet(E.T @ E / n)
        q = m * (2 + m * k)
        rows.append({"k": k, "bic": n * logdet + q * np.log(n)})
    return pd.DataFrame(rows)

@dataclass
class VECMSystem:
    """Full fitted VECM for one window."""

    names: list[str]
    k: int
    n_obs: int
    const: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    resid: np.ndarray

def fit_system(
    x: pd.DataFrame, k: int, names: list[str], keep: np.ndarray | None = None
) -> VECMSystem:
    """One-shot OLS fit of the full VECM system (all equations share the design)."""
    price_cols = [c for c in x.columns if c != "z"]
    P = x[price_cols].to_numpy(float)
    z = x["z"].to_numpy(float)
    m = P.shape[1]
    y, X = _design(P, z, k, k)
    if keep is not None:
        y, X = y[keep], X[keep]
    n = len(y)
    B, *_ = np.linalg.lstsq(X, y, rcond=None)
    E = y - X @ B
    gamma = np.stack([B[2 + i * m : 2 + (i + 1) * m, :].T for i in range(k)])
    return VECMSystem(
        names=list(names), k=k, n_obs=n,
        const=B[0].copy(), alpha=B[1].copy(), gamma=gamma,
        omega=E.T @ E / n, resid=E,
    )

def estimate_window(
    x: pd.DataFrame,
    k: int,
    frame: str,
    names: list[str],
    bar_seconds: float,
    fresh: pd.DataFrame | None = None,
    min_bars: int = 1000,
    stale_filter: bool = False,
) -> VECMWindowResult | None:
    """Estimate one window; None if fewer than ``min_bars`` usable rows."""
    price_cols = [c for c in x.columns if c != "z"]
    P = x[price_cols].to_numpy(float)
    z = x["z"].to_numpy(float)
    y, X = _design(P, z, k, k)

    if stale_filter:
        if fresh is None:
            raise ValueError("stale_filter=True requires the `fresh` frame")
        f = fresh.to_numpy(bool)
        keep = f[k : len(z) - 1].any(axis=1)
        y, X = y[keep], X[keep]

    n = len(y)
    if n < min_bars:
        return None

    L = int(4 * (n / 100) ** (2 / 9))
    m = len(names)
    alpha = np.empty(m)
    alpha_se = np.empty(m)
    alpha_t = np.empty(m)
    lb_p = np.empty(m)
    for j in range(m):
        fit = sm.OLS(y[:, j], X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
        alpha[j], alpha_se[j] = fit.params[1], fit.bse[1]
        alpha_t[j] = fit.tvalues[1]
        lb_p[j] = acorr_ljungbox(fit.resid, lags=[LJUNGBOX_LAGS])["lb_pvalue"].iloc[0]

    beta_alpha = float(alpha[0] - alpha[1:].sum())
    date = x.index[0].date() if len(x) else None
    fresh_share = (
        fresh.mean().to_numpy(float) if fresh is not None else np.full(m, np.nan)
    )
    return VECMWindowResult(
        date=date, frame=frame, k=k, n_obs=n, names=list(names),
        alpha=alpha, alpha_se=alpha_se, alpha_t=alpha_t,
        beta_alpha=beta_alpha,
        half_life_s=half_life_seconds(beta_alpha, bar_seconds),
        ljungbox_p=lb_p, fresh_share=fresh_share,
    )

def _flatten(r: VECMWindowResult) -> dict:
    row = {
        "date": r.date, "frame": r.frame, "k": r.k, "n_obs": r.n_obs,
        "beta_alpha": r.beta_alpha, "half_life_s": r.half_life_s,
    }
    for j, name in enumerate(r.names):
        row[f"alpha_{name}"] = r.alpha[j]
        row[f"alpha_se_{name}"] = r.alpha_se[j]
        row[f"alpha_t_{name}"] = r.alpha_t[j]
        row[f"lb_p_{name}"] = r.ljungbox_p[j]
        row[f"fresh_{name}"] = r.fresh_share[j]
    return row

MIN_UPDATES = 200   # default activity floor: genuine quote updates per leg per window

def min_leg_updates(panel: pd.DataFrame, triangle: Triangle, bar_seconds: float) -> int:
    """Fewest bars in which a leg genuinely updated (quote age <= one bar), over the legs.

    LOCF fills every grid bar, so a bar count says nothing about activity: a no-trading
    holiday still yields 36,000 one-second bars. This counts real quote updates instead.
    """
    if panel.empty:
        return 0
    return int(min((panel[f"age_{s}"] <= bar_seconds).sum() for s in triangle.legs))

def window_fresh_flags(
    panel: pd.DataFrame, triangle: Triangle, frame: str, bar_seconds: float
) -> pd.DataFrame:
    """Fresh-bar flags per series of a frame, from the panel's age_* columns."""
    c, l1, l2 = triangle.cross, triangle.leg1, triangle.leg2
    fresh_leg = {s: panel[f"age_{s}"] <= bar_seconds for s in (c, l1, l2)}
    if frame == TRIVARIATE:
        return pd.DataFrame({s: fresh_leg[s] for s in (c, l1, l2)})
    return pd.DataFrame({c: fresh_leg[c], "synth": fresh_leg[l1] & fresh_leg[l2]})

def fit_windows(
    triangle: Triangle,
    dates: list[dt.date],
    freq: str,
    k: int,
    frame: str,
    liquid: tuple[int, int] = (7, 17),
    stale_filter: bool = False,
    min_bars: int = 1000,
    min_updates: int = MIN_UPDATES,
) -> pd.DataFrame:
    """Estimate per (day x liquid-hours) window; one flattened row per survivor.

    ``min_updates``: every leg must post at least this many genuine quote updates in the
    window (dead-day guard -- see ``min_leg_updates``).
    """
    from ..data.processed import get_panel

    bar_seconds = pd.Timedelta(freq).total_seconds()
    names = frame_names(triangle, frame)
    rows, skipped = [], 0
    for d in dates:
        try:
            panel = get_panel(triangle, d, freq, with_age=True)
        except FileNotFoundError:
            skipped += 1
            continue
        h = panel.index.hour
        win = panel[(h >= liquid[0]) & (h < liquid[1])]
        if win.empty or min_leg_updates(win, triangle, bar_seconds) < min_updates:
            skipped += 1
            continue
        x = make_frame(win, triangle, frame)
        fresh = window_fresh_flags(win, triangle, frame, bar_seconds)
        r = estimate_window(
            x, k, frame, names, bar_seconds,
            fresh=fresh, min_bars=min_bars, stale_filter=stale_filter,
        )
        if r is None:
            skipped += 1
            continue
        rows.append(_flatten(r))
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out

def aggregate_windows(res: pd.DataFrame) -> pd.DataFrame:
    """Across-window summary per coefficient: median, IQR, |t|>1.96 share."""
    names = [c.removeprefix("alpha_") for c in res.columns
             if c.startswith("alpha_") and not c.startswith(("alpha_se_", "alpha_t_"))]
    rows = []
    for name in names:
        a, t = res[f"alpha_{name}"], res[f"alpha_t_{name}"]
        rows.append({
            "series": name,
            "alpha_median": a.median(),
            "alpha_q25": a.quantile(0.25),
            "alpha_q75": a.quantile(0.75),
            "share_|t|>1.96": (t.abs() > 1.96).mean(),
            "lb_pass_rate": (res[f"lb_p_{name}"] > 0.05).mean(),
            "n_windows": len(res),
        })
    return pd.DataFrame(rows).set_index("series")
