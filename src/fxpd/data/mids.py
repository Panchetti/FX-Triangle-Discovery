"""Mid-price construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

def logmid_series(df: pd.DataFrame) -> pd.Series:
    """Log-mid indexed by tick timestamp (UTC), ready for previous-tick sampling."""
    s = np.log(0.5 * (df["bid"] + df["ask"]))
    s.index = pd.DatetimeIndex(df["ts"])
    s.name = "logmid"
    return s
