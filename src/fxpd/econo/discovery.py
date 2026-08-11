"""Gonzalo-Granger component shares and Hasbrouck information shares."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Triangle
from . import vecm

BETA = np.array([1.0, -1.0])       # bivariate cointegrating vector (cross, synth)
BETA_PERP = np.array([1.0, 1.0])

def alpha_perp(alpha: np.ndarray) -> np.ndarray:
    """Orthogonal complement of alpha for the 2-var frame: (alpha_s, -alpha_c)."""
    return np.array([alpha[1], -alpha[0]])

def gg_shares(alpha: np.ndarray, eps: float = 1e-12) -> dict:
    """Gonzalo-Granger component shares {"cross", "synth"}."""
    denom = alpha[1] - alpha[0]
    if abs(denom) < eps:
        return {"cross": np.nan, "synth": np.nan}
    return {"cross": alpha[1] / denom, "synth": -alpha[0] / denom}

def information_shares(alpha: np.ndarray, omega: np.ndarray) -> dict:
    """Hasbrouck IS bounds per market over both Cholesky orderings."""
    r_denom = np.sqrt(omega[0, 0] * omega[1, 1])
    resid_corr = float(omega[0, 1] / r_denom) if r_denom > 0 else np.nan

    psi = alpha_perp(alpha)
    den = float(psi @ omega @ psi)
    nan = {"cross": (np.nan, np.nan), "synth": (np.nan, np.nan), "resid_corr": resid_corr}
    if den <= 0:
        return nan
    try:
        L1 = np.linalg.cholesky(omega)
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        L2 = np.linalg.cholesky(P @ omega @ P.T)
    except np.linalg.LinAlgError:
        return nan
    is1 = (psi @ L1) ** 2 / den
    is2 = (((psi @ P) @ L2) ** 2 / den)[::-1]
    return {
        "cross": (float(min(is1[0], is2[0])), float(max(is1[0], is2[0]))),
        "synth": (float(min(is1[1], is2[1])), float(max(is1[1], is2[1]))),
        "resid_corr": resid_corr,
    }

def long_run_impact(alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Closed-form Hasbrouck long-run impact Psi(1) = b_perp (a_perp' G(1) b_perp)^-1 a_perp'."""
    ap = alpha_perp(alpha)
    gamma1 = np.eye(len(alpha)) - gamma.sum(axis=0)
    scalar = ap @ gamma1 @ BETA_PERP
    return np.outer(BETA_PERP, ap) / scalar

def psi_companion(
    alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray = BETA, horizon: int = 2000
) -> np.ndarray:
    """Psi(1) via the levels-VAR(k+1) impulse response at a long horizon."""
    m, k = len(alpha), gamma.shape[0]
    I = np.eye(m)
    A = [I + np.outer(alpha, beta) + gamma[0]]
    A += [gamma[i - 1] - gamma[i - 2] for i in range(2, k + 1)]
    A.append(-gamma[k - 1])
    theta = [I]
    for j in range(1, horizon + 1):
        acc = np.zeros((m, m))
        for i in range(1, min(j, k + 1) + 1):
            acc = acc + A[i - 1] @ theta[j - i]
        theta.append(acc)
    return theta[-1]

@dataclass
class DiscoveryWindowResult:
    date: dt.date
    k: int
    n_obs: int
    gg_cross: float
    gg_synth: float
    is_cross_lo: float
    is_cross_hi: float
    is_synth_lo: float
    is_synth_hi: float
    resid_corr: float
    half_life_s: float
    half_life_dyn_s: float
    valid: bool

def discovery_window(system: vecm.VECMSystem, bar_seconds: float) -> DiscoveryWindowResult:
    """GG + IS bounds + half-life for one fitted bivariate VECM system."""
    alpha, omega = system.alpha, system.omega
    gg = gg_shares(alpha)
    is_ = information_shares(alpha, omega)
    beta_alpha = float(alpha[0] - alpha[1])
    valid = bool(np.isfinite(gg["cross"]) and 0.0 <= gg["cross"] <= 1.0)
    return DiscoveryWindowResult(
        date=None, k=system.k, n_obs=system.n_obs,
        gg_cross=gg["cross"], gg_synth=gg["synth"],
        is_cross_lo=is_["cross"][0], is_cross_hi=is_["cross"][1],
        is_synth_lo=is_["synth"][0], is_synth_hi=is_["synth"][1],
        resid_corr=is_["resid_corr"],
        half_life_s=vecm.half_life_seconds(beta_alpha, bar_seconds),
        half_life_dyn_s=vecm.dynamic_half_life(
            system.alpha, system.gamma, BETA, bar_seconds),
        valid=valid,
    )

def _window_slice(panel: pd.DataFrame, liquid, session) -> pd.DataFrame:
    """Select a day's estimation window: liquid hours, further sliced by session if given."""
    h = panel.index.hour
    liq = (h >= liquid[0]) & (h < liquid[1])
    if session is not None:
        from ..calendar import sessions

        sess = np.asarray(sessions.label_sessions(panel.index) == session)
        return panel[np.asarray(liq) & sess]
    return panel[liq]

def fit_discovery_windows(
    triangle: Triangle,
    dates: list[dt.date],
    freq: str,
    k: int,
    liquid: tuple[int, int] = (7, 17),
    min_bars: int = 1000,
    session: str | None = None,
    stale_filter: bool = False,
    bar_mask: pd.Series | None = None,
    mask_tolerance: str = "30min",
) -> pd.DataFrame:
    """Per (day x window) bivariate discovery; one row per estimated window."""
    from ..data.processed import get_panel

    bar_seconds = pd.Timedelta(freq).total_seconds()
    names = vecm.frame_names(triangle, vecm.BIVARIATE)
    rows, skipped = [], 0
    for d in dates:
        try:
            panel = get_panel(triangle, d, freq, with_age=True)
        except FileNotFoundError:
            skipped += 1
            continue
        win = _window_slice(panel, liquid, session)
        x = vecm.make_frame(win, triangle, vecm.BIVARIATE)
        keep = None
        if stale_filter:
            fresh = vecm.window_fresh_flags(win, triangle, vecm.BIVARIATE, bar_seconds)
            all_fresh = fresh.all(axis=1).to_numpy()
            keep = all_fresh[k : len(x) - 1]
        if bar_mask is not None:
            # regime mask: a row survives only if its whole dependency window is in-state
            m = (bar_mask.reindex(win.index, method="ffill",
                                  tolerance=pd.Timedelta(mask_tolerance))
                 .fillna(False).astype(bool))
            ok = m.rolling(k + 2).min().fillna(0).astype(bool).to_numpy()
            in_regime = ok[k + 1 : len(x)]
            keep = in_regime if keep is None else (keep & in_regime)
        n_obs = int(keep.sum()) if keep is not None else len(x) - (k + 1)
        if n_obs < min_bars:
            skipped += 1
            continue
        system = vecm.fit_system(x, k, names, keep=keep)
        r = discovery_window(system, bar_seconds)
        r.date = d
        rows.append(vars(r))
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out

def fit_discovery_windows_refresh(
    triangle: Triangle,
    dates: list[dt.date],
    k: int,
    liquid: tuple[int, int] = (7, 17),
    min_bars: int = 1000,
    session: str | None = None,
) -> pd.DataFrame:
    """Per (day x window) bivariate discovery sampled in refresh time."""
    from ..data.refresh import mean_refresh_seconds, refresh_panel

    names = vecm.frame_names(triangle, vecm.BIVARIATE)
    rows, skipped = [], 0
    for d in dates:
        try:
            panel = refresh_panel(triangle, d, liquid=liquid)
        except FileNotFoundError:
            skipped += 1
            continue
        win = _window_slice(panel, liquid, session)
        x = vecm.make_frame(win, triangle, vecm.BIVARIATE)
        if len(x) - (k + 1) < min_bars:
            skipped += 1
            continue
        bar_seconds = mean_refresh_seconds(win.index)
        system = vecm.fit_system(x, k, names)
        r = discovery_window(system, bar_seconds)
        r.date = d
        rows.append(vars(r))
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out

def aggregate_discovery(res: pd.DataFrame) -> pd.DataFrame:
    """Across-window summary over valid windows: GG median [IQR], IS [lo, hi]."""
    n = len(res)
    valid = res[res["valid"]] if "valid" in res.columns else res.iloc[:0]
    gg_incl = res["gg_cross"].replace([np.inf, -np.inf], np.nan).dropna() \
        if "gg_cross" in res.columns else pd.Series(dtype=float)
    incl = {
        "gg_inclusive_median": float(gg_incl.median()) if len(gg_incl) else np.nan,
        "gg_inclusive_q25": float(gg_incl.quantile(0.25)) if len(gg_incl) else np.nan,
        "gg_inclusive_q75": float(gg_incl.quantile(0.75)) if len(gg_incl) else np.nan,
        "n_gg_inclusive": int(len(gg_incl)),
    }
    cols = ["gg_median", "gg_q25", "gg_q75", "is_lo", "is_hi"]
    if len(valid) == 0:
        out = pd.DataFrame(np.nan, index=pd.Index(["cross", "synth"], name="series"), columns=cols)
        out.attrs.update(n_windows=n, n_valid=0, exclusion_rate=(1.0 if n else np.nan),
                         resid_corr_median=np.nan, half_life_median=np.nan,
                         n_is_valid={"cross": 0, "synth": 0}, **incl)
        return out
    rows, n_is_valid = {}, {}
    for series in ("cross", "synth"):
        gg = valid[f"gg_{series}"]
        n_is_valid[series] = int(valid[f"is_{series}_lo"].notna().sum())
        rows[series] = {
            "gg_median": gg.median(),
            "gg_q25": gg.quantile(0.25),
            "gg_q75": gg.quantile(0.75),
            "is_lo": valid[f"is_{series}_lo"].median(),
            "is_hi": valid[f"is_{series}_hi"].median(),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "series"
    out.attrs.update(
        n_windows=n,
        n_valid=int(len(valid)),
        exclusion_rate=float(1 - len(valid) / n) if n else np.nan,
        resid_corr_median=float(valid["resid_corr"].median()),
        half_life_median=float(valid["half_life_s"].median()),
        half_life_dyn_median=(float(valid["half_life_dyn_s"].median())
                              if "half_life_dyn_s" in valid else np.nan),
        n_is_valid=n_is_valid,
        **incl,
    )
    return out

def select_k_for_freq(
    triangle: Triangle,
    dates: list[dt.date],
    freq: str,
    liquid: tuple[int, int] = (7, 17),
    n_windows: int = 6,
    k_max: int = 20,
) -> int:
    """Modal BIC-argmin k over the first few liquid-hours windows at ``freq``."""
    from ..data.processed import get_panel

    ks = []
    for d in dates:
        if len(ks) >= n_windows:
            break
        try:
            panel = get_panel(triangle, d, freq, with_age=True)
        except FileNotFoundError:
            continue
        x = vecm.make_frame(_liquid_slice(panel, liquid), triangle, vecm.BIVARIATE)
        if len(x) < 5000:
            continue
        bic = vecm.select_k_bic(x, k_max=k_max)
        ks.append(int(bic.loc[bic["bic"].idxmin(), "k"]))
    if not ks:
        raise FileNotFoundError(f"No usable windows to select k at {freq}")
    return int(pd.Series(ks).mode().iloc[0])

def frequency_curve(
    triangle: Triangle,
    dates: list[dt.date],
    freqs: list[str],
    k: int | dict | None = None,
    liquid: tuple[int, int] = (7, 17),
) -> pd.DataFrame:
    """Median GG + IS bounds per frequency."""
    rows = []
    for freq in freqs:
        if k is None:
            k_used = select_k_for_freq(triangle, dates, freq, liquid=liquid)
        elif isinstance(k, dict):
            k_used = k[freq]
        else:
            k_used = k
        res = fit_discovery_windows(triangle, dates, freq, k_used, liquid=liquid)
        agg = aggregate_discovery(res)
        for series in ("cross", "synth"):
            rows.append({
                "freq": freq, "series": series,
                "gg_median": agg.loc[series, "gg_median"],
                "is_lo": agg.loc[series, "is_lo"],
                "is_hi": agg.loc[series, "is_hi"],
                "resid_corr": agg.attrs["resid_corr_median"],
                "k_used": k_used,
                "n_windows": agg.attrs["n_valid"],
                "exclusion_rate": agg.attrs["exclusion_rate"],
            })
    return pd.DataFrame(rows)
