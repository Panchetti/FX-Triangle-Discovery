"""2-state Gaussian HMM on log realized volatility, filtered probabilities only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

HI, LO = 0.7, 0.3

# NB: a fixed TRAIN-split standardiser was tried first and classified 100% of EVAL as
# "stressed" (non-stationary level across the chronological split) -- hence the causal
# trailing standardisation below.
def rolling_standardise(
    x: np.ndarray, window: int = 400, min_periods: int = 100, clip: float = 5.0
) -> np.ndarray:
    """Trailing z-score of x against its own past (stats shifted one step back)."""
    s = pd.Series(np.asarray(x, float))
    mu = s.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = s.rolling(window, min_periods=min_periods).std().shift(1)
    z = (s - mu) / sd.replace(0, np.nan)
    return (z.clip(-clip, clip) if clip is not None else z).to_numpy()

def fit_hmm(x_train: np.ndarray, lengths: list[int] | None = None, seed: int = 0,
            min_covar: float = 1e-2) -> GaussianHMM:
    x = np.asarray(x_train, float).reshape(-1, 1)
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200,
                        random_state=seed, min_covar=min_covar)
    model.fit(x, lengths)   # lengths marks day boundaries -> no transition spans a gap
    return model

def hmm_params(model: GaussianHMM) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(model.startprob_, float),
        np.asarray(model.transmat_, float),
        np.asarray(model.means_, float).ravel(),
        np.asarray(model.covars_, float).ravel(),
    )

def forward_filter(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Filtered P(s_t | x_1..t), (T, K) -- uses only data up to t, so it is causal."""
    x = np.asarray(x, float).ravel()
    mu = np.asarray(means, float).ravel()
    var = np.asarray(variances, float).ravel()
    T, K = x.size, mu.size
    B = np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]) / np.sqrt(
        2 * np.pi * var[None, :]
    )
    alpha = np.empty((T, K))
    v = np.asarray(startprob, float) * B[0]
    alpha[0] = v / v.sum()
    for t in range(1, T):
        v = (alpha[t - 1] @ transmat) * B[t]
        s = v.sum()
        alpha[t] = v / s if s > 0 else alpha[t - 1]
    return alpha

def filtered_by_segment(
    x: np.ndarray, lengths: list[int], startprob, transmat, means, variances
) -> np.ndarray:
    """Filter each day independently so states don't bleed across the overnight gap."""
    out, i = [], 0
    for n in lengths:
        out.append(forward_filter(x[i : i + n], startprob, transmat, means, variances))
        i += n
    return np.vstack(out)

def stressed_index(means: np.ndarray) -> int:
    # stressed = larger-mean state; EM's own state ordering is arbitrary
    return int(np.argmax(np.asarray(means, float).ravel()))

def label_states(p_stressed: np.ndarray, hi: float = HI, lo: float = LO) -> np.ndarray:
    p = np.asarray(p_stressed, float)
    out = np.full(p.shape, "uncertain", dtype=object)
    out[p > hi] = "stressed"
    out[p < lo] = "calm"
    return out

def lookahead_ledger(n_train: int, n_eval: int, standardiser_desc: str,
                     model: GaussianHMM) -> dict:
    mu, var = hmm_params(model)[2], hmm_params(model)[3]
    return {
        "blocks_train (EM fit here)": n_train,
        "blocks_eval (ALL reported results here)": n_eval,
        "standardisation": standardiser_desc,
        "fitted_means (standardised)": np.round(mu, 4).tolist(),
        "fitted_sds": np.round(np.sqrt(var), 4).tolist(),
        "stressed_state_index": stressed_index(mu),
        "posterior_used": "FILTERED (hand forward filter) -- predict_proba never called",
    }

def lag_state_within_day(states, dates) -> pd.Series:
    """Use the previous block's regime label within each day."""
    df = pd.DataFrame({"state": np.asarray(states, dtype=object), "date": np.asarray(dates)})
    return df.groupby("date")["state"].shift(1)

def state_count_bic(x_train: np.ndarray, lengths=None, ks=(1, 2, 3), seed: int = 0) -> pd.DataFrame:
    """BIC across K-state fits; two states are imposed for interpretability, not selected."""
    x = np.asarray(x_train, float).reshape(-1, 1)
    n, rows = x.shape[0], []
    for K in ks:
        if K == 1:
            mu, sd = float(x.mean()), float(x.std())
            ll = float(np.sum(-0.5 * np.log(2 * np.pi * sd**2) - 0.5 * ((x - mu) / sd) ** 2))
            q = 2
        else:
            mk = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200,
                             random_state=seed, min_covar=1e-2).fit(x, lengths)
            ll, q = float(mk.score(x, lengths)), K - 1 + K * (K - 1) + 2 * K
        rows.append({"K": K, "loglik": ll, "n_params": q, "bic": -2 * ll + q * np.log(n)})
    return pd.DataFrame(rows)
