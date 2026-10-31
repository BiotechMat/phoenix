"""P5.3 tests: residual ridge, forgetting, abstention threshold, outlier guard, store wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.contracts import DataAvailability
from core.models.online_residual import OnlineResidualRidge, abstention_threshold, outlier_guard_skip
from tests.phase0_helpers import build_store, companion_flat_bars, make_snapshot, ohlc_path, sine_history
from ui.api_server import MIN_TRAIN_LABELLED_BARS


def test_ridge_learns_a_linear_residual_and_clips_probabilities() -> None:
    i = np.arange(400, dtype=np.float64)
    x = np.column_stack([np.sin(i / 7.0), np.cos(i / 5.0)])
    residual = 0.1 * x[:, 0] - 0.05
    ridge = OnlineResidualRidge(2, l2=0.0001, forgetting=1.0).fit(x, residual)
    adj = ridge.adjust(np.full(400, 0.5), x)
    assert np.allclose(adj, 0.5 + residual, atol=1e-3)
    assert ridge.adjust(np.array([0.99995]), np.array([[10.0, 0.0]]))[0] <= 1.0 - 1e-4


def test_forgetting_favours_recent_rows_and_warm_start_matches_batch() -> None:
    x = np.ones((200, 1)) * np.linspace(-1, 1, 200)[:, None]
    old = 0.2 * x[:100, 0]
    new = -0.2 * x[100:, 0]
    batch = OnlineResidualRidge(1, l2=1e-6, forgetting=0.95).fit(x, np.concatenate([old, new]))
    assert batch.coef[1] < 0.0  # the regime of the newest rows dominates
    warm = OnlineResidualRidge(1, l2=1e-6, forgetting=0.95).fit(x, np.concatenate([old, new]))
    split = OnlineResidualRidge(1, l2=1e-6, forgetting=0.95)
    split.fit(x, np.concatenate([old, new]))  # same standardisation, then compare incremental update
    inc = OnlineResidualRidge(1, l2=1e-6, forgetting=0.95)
    inc._mean, inc._std = split._mean.copy(), split._std.copy()
    inc.update(x[:100], old).update(x[100:], new)
    assert np.allclose(inc.coef, warm.coef, atol=1e-8)
    assert inc.n_seen == 200


def test_abstention_threshold_and_outlier_guard() -> None:
    di = np.linspace(0.0, 10.0, 1001)
    assert abstention_threshold(di) == pytest.approx(9.9)
    assert abstention_threshold(np.array([np.nan])) == float("inf")
    skip, frac = outlier_guard_skip(np.array([1.0, 2.0, 5.0, 6.0]), gate_threshold=4.0)
    assert skip and frac == 0.5
    assert outlier_guard_skip(np.array([1.0, 2.0, 3.0, 5.0]), gate_threshold=4.0) == (False, 0.25)
    assert outlier_guard_skip(np.array([]), 4.0) == (False, 0.0)


def test_store_fits_residual_layer_and_reports_abstention(tmp_path: Path) -> None:
    history = sine_history("BTC", MIN_TRAIN_LABELLED_BARS + 120, 100.0)
    live = ohlc_path("BTC", [(100.0, 100.3, 99.7, 100.2), (100.2, 100.5, 99.9, 100.1)])
    store = build_store(
        tmp_path,
        live_bars={"BTC": live, **companion_flat_bars(2)},
        snapshots=[make_snapshot()],
        history={"BTC": history},
        availability=DataAvailability(candles=True, funding_history=True, open_interest_history=True, feed_source="SCRIPTED"),
    )
    assert store.ml_worker is not None
    tele = store.residual_telemetry
    assert tele["residual_layer"] is True and tele["abstention_di_threshold"] is not None
    store.step_simulation_tick()
    tele = store.residual_telemetry
    assert tele["predictions"] == 1 and 0.0 <= tele["abstention_rate"] <= 1.0
    assert 0.0 < store.state.meta_predictions[-1].p_meta < 1.0
