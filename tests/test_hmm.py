"""Filtered-only HMM (Week 6). The invariance test is the week's best test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fxpd.regimes.hmm import (
    fit_hmm,
    forward_filter,
    hmm_params,
    label_states,
    stressed_index,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "fxpd" / "regimes"


def _two_state_series(T=600, seed=0):
    rng = np.random.default_rng(seed)
    state = np.zeros(T, int)
    for t in range(1, T):
        state[t] = state[t - 1] if rng.random() < 0.95 else 1 - state[t - 1]
    x = rng.normal(0, 1, T) + 3.0 * state
    return x, state


def _fitted(x):
    m = fit_hmm(x, seed=0)
    return hmm_params(m)


def test_forward_filter_invariance_exact():
    """filtered(x[:t])[-1] MUST equal filtered(x)[t-1] -- the lookahead guarantee."""
    x, _ = _two_state_series()
    sp, tm, mu, var = _fitted(x)
    full = forward_filter(x, sp, tm, mu, var)
    rng = np.random.default_rng(1)
    for t in rng.integers(20, len(x), size=6):
        trunc = forward_filter(x[:t], sp, tm, mu, var)
        assert np.max(np.abs(trunc[-1] - full[t - 1])) < 1e-12


def test_smoothed_predict_proba_FAILS_the_same_check():
    """Documents WHY the hand filter exists: hmmlearn's posteriors use the future."""
    x, _ = _two_state_series()
    m = fit_hmm(x, seed=0)
    cut = 400
    full = m.predict_proba(x.reshape(-1, 1))
    trunc = m.predict_proba(x[:cut].reshape(-1, 1))
    assert np.max(np.abs(trunc[-1] - full[cut - 1])) > 0.1  # smoothed -> not invariant


def test_filtered_recovers_true_state():
    x, state = _two_state_series()
    sp, tm, mu, var = _fitted(x)
    p = forward_filter(x, sp, tm, mu, var)[:, stressed_index(mu)]
    assert (( p > 0.5).astype(int) == state).mean() > 0.9


def test_stressed_index_and_thresholds():
    assert stressed_index(np.array([0.2, 1.9])) == 1
    assert stressed_index(np.array([1.9, 0.2])) == 0      # never trust EM ordering
    lab = label_states(np.array([0.95, 0.5, 0.05]))
    assert list(lab) == ["stressed", "uncertain", "calm"]


def test_no_smoothed_calls_in_regimes_source():
    """regimes/ must never CALL predict_proba / score_samples.

    AST-based (not a string grep) so that documenting the ban in a docstring does not
    trip the guard -- we check for actual method calls.
    """
    import ast

    banned = {"predict_proba", "score_samples"}
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in banned, f"{path.name} calls {node.func.attr}"
