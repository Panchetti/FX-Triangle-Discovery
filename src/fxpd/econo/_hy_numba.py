"""Hayashi-Yoshida covariation + lead-lag: numba-njit tier."""

from __future__ import annotations

import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def hy_cov(tx, x, ty, y):
    rx = x[1:] - x[:-1]
    ry = y[1:] - y[:-1]
    ax, bx = tx[:-1], tx[1:]
    ay, by = ty[:-1], ty[1:]
    i = 0
    j = 0
    n = rx.shape[0]
    m = ry.shape[0]
    cov = 0.0
    while i < n and j < m:
        if bx[i] > ay[j] and by[j] > ax[i]:
            cov += rx[i] * ry[j]
        if bx[i] <= by[j]:
            i += 1
        else:
            j += 1
    return cov

@njit(cache=True, fastmath=False)
def hy_leadlag(tx, x, ty, y, thetas_ns):
    rvx = np.sum((x[1:] - x[:-1]) ** 2)
    rvy = np.sum((y[1:] - y[:-1]) ** 2)
    denom = np.sqrt(rvx * rvy)
    out = np.empty(thetas_ns.shape[0], dtype=np.float64)
    for k in range(thetas_ns.shape[0]):
        out[k] = hy_cov(tx, x, ty + thetas_ns[k], y) / denom
    return out
