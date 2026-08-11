"""Week-1 sanity figure: the basis over time, its distribution, and the legs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Triangle
from .checks import BPS, basis_stats

def plot_sanity(
    panel: pd.DataFrame, triangle: Triangle, out_path: Path | str | None = None
) -> plt.Figure:
    """Three-panel diagnostic; returns the Figure (and saves a PNG if out_path given)."""
    z_bps = panel["z"] * BPS
    stats = basis_stats(panel["z"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    ax = axes[0]
    ax.plot(panel.index, z_bps, lw=0.5, color="tab:blue")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_title(
        f"[{triangle.name}] basis  z = {triangle.cross} - {triangle.leg1} - {triangle.leg2}"
        f"   (mean {stats['mean_bps']:.3f} bp, std {stats['std_bps']:.3f} bp, "
        f"|max| {stats['abs_max_bps']:.2f} bp)"
    )
    ax.set_ylabel("z  (bps)")

    ax = axes[1]
    ax.hist(z_bps, bins=200, color="tab:blue", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(
        f"basis distribution  (excess kurtosis {stats['excess_kurtosis']:.1f}; "
        f"{100 * stats['frac_gt_1bp']:.1f}% of bars exceed 1 bp)"
    )
    ax.set_xlabel("z  (bps)")
    ax.set_ylabel("count (log)")

    ax = axes[2]
    for leg in triangle.legs:
        demeaned = panel[leg] - panel[leg].mean()
        ax.plot(panel.index, demeaned, lw=0.6, label=leg)
    ax.set_title("legs (log-mid, demeaned) — they ride a common efficient price")
    ax.set_ylabel("log-mid − mean")
    ax.legend(loc="upper left", ncol=3)

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130)
    return fig
