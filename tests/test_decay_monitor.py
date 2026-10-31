"""P5.2 tests: rolling IC, sign-aligned breach rule, half-lives and cadence, early retrain trigger in the store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.decay_monitor import DecayConfig, evaluate_decay
from tests.phase0_helpers import build_store, companion_flat_bars, flat_bars, make_snapshot, sine_history

H = 4


def _series(n: int = 2400, flip_at: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Closes whose next-4-bar return is driven by a feature (and by its negative after ``flip_at``)."""
    i = np.arange(n, dtype=np.float64)
    feat = np.sin(i * 0.37) + 0.5 * np.cos(i * 0.11)
    drive = feat.copy()
    if flip_at is not None:
        drive[flip_at:] *= -1.0
    rets = np.zeros(n)
    rets[H:] = 0.001 * drive[:-H] / H  # return over the next H bars follows the feature
    closes = 100.0 * np.exp(np.cumsum(rets))
    return feat, closes


def test_stable_feature_does_not_breach() -> None:
    feat, closes = _series()
    rep = evaluate_decay({"f": feat}, closes, DecayConfig(horizon_bars=H))
    f = rep.features["f"]
    assert f.long_run_ic is not None and f.long_run_ic > 0.5
    assert not f.breached and not rep.retrain_triggered
    assert all(v > 0.5 for v in f.aligned_rolling_ic)


def test_flipped_feature_breaches_and_triggers_retrain() -> None:
    feat, closes = _series(flip_at=2000)
    rep = evaluate_decay({"f": feat}, closes, DecayConfig(horizon_bars=H))
    assert rep.features["f"].breached and rep.retrain_triggered and rep.breached_features == ["f"]
    assert rep.features["f"].aligned_rolling_ic[-1] < 0.0


def test_half_lives_and_cadence_are_reported() -> None:
    feat, closes = _series()
    rep = evaluate_decay({"f": feat}, closes, DecayConfig(horizon_bars=H))
    assert rep.recommended_cadence_bars == rep.features["f"].half_life_bars
    assert rep.recommended_cadence_bars is None or rep.recommended_cadence_bars > 0.0


def test_store_retrains_early_on_breach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 2, 100.0), **companion_flat_bars(2)},
        snapshots=[make_snapshot()],
        history={"BTC": sine_history("BTC", 450, 100.0)},
    )
    calls: list[int] = []

    def _fake_fit(bars: object) -> None:
        calls.append(1)
        store._last_retrain_message = "fit attempted"
        return None

    monkeypatch.setattr(store, "_fit_worker_on_bars", _fake_fit)
    monkeypatch.setattr(store, "_check_decay", lambda ts: True)
    store.step_simulation_tick()  # well before the daily schedule, but the breach forces a retrain
    assert calls == [1]
    assert any("DECAY" in line for line in store._activity_log)

    monkeypatch.setattr(store, "_check_decay", lambda ts: False)
    store.step_simulation_tick()
    assert calls == [1]  # no breach, not yet a day: no retrain


def test_store_reports_decay_after_enough_history(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0), **companion_flat_bars(1)},
        snapshots=[make_snapshot()],
        history={"BTC": sine_history("BTC", 450, 100.0)},
    )
    assert store.decay_telemetry["available"] is False
    store.step_simulation_tick()
    tele = store.decay_telemetry
    assert tele["available"] is True and set(tele["features"]) == {"ffd_close_zscore", "ffd_oi_zscore", "oi_funding_basis_zscore"}
