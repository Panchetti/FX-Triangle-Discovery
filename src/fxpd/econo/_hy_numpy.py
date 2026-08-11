"""Hayashi-Yoshida covariation + lead-lag: pure-numpy reference."""

from __future__ import annotations

import numpy as np

def hy_cov(tx: np.ndarray, x: np.ndarray, ty: np.ndarray, y: np.ndarray) -> float:
    """Hayashi-Yoshida covariation over half-open interval overlaps, O(n+m)."""
    rx = x[1:] - x[:-1]
    ry = y[1:] - y[:-1]
    ax, bx = tx[:-1], tx[1:]
    ay, by = ty[:-1], ty[1:]
    i = j = 0
    n, m = rx.shape[0], ry.shape[0]
    cov = 0.0
    while i < n and j < m:
        if bx[i] > ay[j] and by[j] > ax[i]:
            cov += rx[i] * ry[j]
        if bx[i] <= by[j]:                       # advance the interval that ends first
            i += 1
        else:
            j += 1
    return float(cov)

def hy_leadlag(
    tx: np.ndarray, x: np.ndarray, ty: np.ndarray, y: np.ndarray, thetas_ns: np.ndarray
) -> np.ndarray:
    """rho(theta) = HY-corr(X, Y shifted by +theta) for each theta (nanoseconds)."""
    rvx = np.sum((x[1:] - x[:-1]) ** 2)
    rvy = np.sum((y[1:] - y[:-1]) ** 2)
    denom = np.sqrt(rvx * rvy)
    out = np.empty(thetas_ns.shape[0], dtype=np.float64)
    for k in range(thetas_ns.shape[0]):
        out[k] = hy_cov(tx, x, ty + thetas_ns[k], y) / denom
    return out
