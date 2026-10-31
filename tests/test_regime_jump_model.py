"""P3.3 tests: jump model fitting, persistence, causality, and the walk-forward conditioning harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.regime_eval import walk_forward
from research.trial_registry import TrialRegistry
from risk.regime_jump_model import fit_jump_model, mean_dwell


def _regimes(n_blocks: int = 12, block: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Alternating calm (0) and stressed (1) blocks with deterministic within-block wiggle."""
    t = np.arange(n_blocks * block, dtype=np.float64)
    truth = np.repeat(np.arange(n_blocks) % 2, block)
    vol = np.where(truth == 1, 3.0, 0.0) + 0.4 * np.sin(t * 0.7)
    fund = np.where(truth == 1, -1.0, 1.0) + 0.3 * np.cos(t * 1.3)
    return np.column_stack([vol, fund]), truth


def test_fit_recovers_constructed_regimes_and_labels_by_first_feature() -> None:
    x, truth = _regimes()
    m = fit_jump_model(x, 2, jump_penalty=20.0, n_init=3)
    s = m.smooth_states(x)
    assert np.mean(s == truth) > 0.99
    assert m.centroids[0, 0] < m.centroids[1, 0]  # state 0 is the low-volatility state
    assert mean_dwell(s) == pytest.approx(400.0, rel=0.05)


def test_jump_penalty_increases_persistence() -> None:
    x, _ = _regimes()
    noisy = x + 1.2 * np.column_stack([np.sin(np.arange(x.shape[0]) * 2.3), np.cos(np.arange(x.shape[0]) * 3.1)])
    low = fit_jump_model(noisy, 2, jump_penalty=0.0, n_init=3).filter_states(noisy)
    high = fit_jump_model(noisy, 2, jump_penalty=50.0, n_init=3).filter_states(noisy)
    assert mean_dwell(high) > mean_dwell(low)


def test_forward_filter_is_causal() -> None:
    x, _ = _regimes()
    m = fit_jump_model(x, 2, jump_penalty=20.0, n_init=2)
    full = m.filter_states(x)
    prefix = m.filter_states(x[:2000])
    assert np.array_equal(full[:2000], prefix)  # appending the future never changes past states


def test_walk_forward_conditioning_improves_a_regime_dependent_sleeve(tmp_path: Path) -> None:
    x, truth = _regimes(n_blocks=24, block=500)
    ts = np.arange(x.shape[0], dtype=np.int64) * 3_600_000
    # The sleeve earns in calm blocks and loses in stressed ones.
    sleeve = np.where(truth == 0, 0.0004, -0.0006) + 0.0001 * np.sin(np.arange(x.shape[0]) * 0.9)
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    rep = walk_forward(
        ts, x, sleeve, sleeve="synthetic", initial_train_hours=3000, test_hours=1500, validation_hours=1000,
        state_grid=(2, 3), penalty_grid=(10.0, 50.0), registry=reg,
    )
    assert len(rep.folds) >= 5
    assert rep.oof_conditioned > rep.oof_unconditioned and rep.improves
    assert rep.persistent and rep.mean_dwell_hours > 24.0
    assert reg.count("regime") == 1 and rep.trial_id is not None
