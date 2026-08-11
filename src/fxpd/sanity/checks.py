"""Stationarity and magnitude diagnostics for the arbitrage basis z_t."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

BPS = 1e4

def basis_stats(z: pd.Series) -> dict:
    """Magnitude / tail summary of the basis, reported in basis points."""
    zb = z.dropna() * BPS
    return {
        "n": int(zb.size),
        "mean_bps": float(zb.mean()),
        "std_bps": float(zb.std()),
        "mad_bps": float((zb - zb.median()).abs().median()),
        "p01_bps": float(zb.quantile(0.01)),
        "p99_bps": float(zb.quantile(0.99)),
        "min_bps": float(zb.min()),
        "max_bps": float(zb.max()),
        "abs_max_bps": float(zb.abs().max()),
        "excess_kurtosis": float(zb.kurtosis()),
        "frac_gt_1bp": float((zb.abs() > 1.0).mean()),
    }

def adf_kpss(
    z: pd.Series, max_n: int = 40_000, kpss_regression: str = "c"
) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary) on the basis."""
    zz = z.dropna()
    n_full = int(zz.size)
    step = max(1, n_full // max_n)
    zz = zz.iloc[::step].to_numpy()
    n = int(zz.size)
    maxlag = min(int(round(12 * (n / 100) ** 0.25)), 24)

    adf_stat, adf_p, *_ = adfuller(zz, maxlag=maxlag, autolag="AIC")
    with warnings.catch_warnings():
        # KPSS warns outside its p-value table (p<.01 or p>.10); surfaced as-is
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(zz, regression=kpss_regression, nlags="auto")
    return {
        "n_full": n_full,
        "n_used": n,
        "subsample_step": int(step),
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "adf_rejects_unit_root": bool(adf_p < 0.05),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_rejects_stationarity": bool(kpss_p < 0.05),
        "stationary_consensus": bool(adf_p < 0.05 and kpss_p >= 0.05),
    }

def summarize(z: pd.Series) -> dict:
    """Convenience: magnitude stats + stationarity tests in one dict."""
    return {**basis_stats(z), **adf_kpss(z)}
