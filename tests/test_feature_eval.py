"""P1.8 tests: IC statistics, half-life, CPCV log-loss gate, registry logging; noise must fail."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from research.feature_eval import (
    FeatureEvalConfig,
    evaluate_existing_features,
    evaluate_feature,
    forward_log_returns,
    ic_half_life,
    ic_tstat,
    prepare_bars,
    spearman_ic,
)
from research.trial_registry import TrialRegistry
from tests.phase0_helpers import sine_history


def test_forward_returns_and_ic_statistics() -> None:
    closes = np.array([100.0, 101.0, 102.0, 101.0, 103.0])
    fwd = forward_log_returns(closes, 2)
    assert fwd[0] == pytest.approx(math.log(102.0 / 100.0)) and np.isnan(fwd[-1]) and np.isnan(fwd[-2])
    x = np.arange(50, dtype=np.float64)
    assert spearman_ic(x, x * 2.0 + 1.0) == pytest.approx(1.0)
    assert spearman_ic(x, -x) == pytest.approx(-1.0)
    assert spearman_ic(np.ones(50), x) is None  # constant feature has no IC
    assert ic_tstat(0.1, 1000, 1) == pytest.approx(0.1 * math.sqrt(998) / math.sqrt(0.99))
    assert ic_tstat(0.1, 1000, 4) == pytest.approx(0.1 * math.sqrt(248) / math.sqrt(0.99))  # overlap-adjusted


def test_ic_half_life_of_a_feature_built_from_a_persistent_signal() -> None:
    n = 3000
    i = np.arange(n, dtype=np.float64)
    # An AR(1)-like deterministic driver with decay 0.9 per bar; returns load on the driver.
    driver = np.zeros(n)
    shock = np.sin(i * 1.7) + np.cos(i * 0.37)
    for t in range(1, n):
        driver[t] = 0.9 * driver[t - 1] + shock[t]
    rets = 0.001 * driver + 0.002 * np.sin(i * 2.9)
    closes = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(rets[:-1]))))
    hl = ic_half_life(driver, closes, max_lag=40)
    assert hl is not None and 2.0 <= hl <= 15.0  # ln(2)/ln(1/0.9) ≈ 6.6 bars


@pytest.fixture(scope="module")
def prepared_bars() -> tuple[list, object]:  # type: ignore[type-arg]
    bars = sine_history("BTC", 900, 100.0, amplitude_pct=0.04, period_bars=96)
    return bars, prepare_bars(bars)


def test_p1_8_noise_feature_fails_and_leaked_signal_passes(prepared_bars: tuple[list, object], tmp_path: Path) -> None:  # type: ignore[type-arg]
    bars, prep = prepared_bars
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    noise = np.random.default_rng(12345).standard_normal(len(bars))  # research control, not a live path
    res_noise = evaluate_feature("noise_control", noise, prepared=prep, registry=reg)  # type: ignore[arg-type]
    assert res_noise.passed is False
    assert res_noise.folds_evaluated >= 5
    assert res_noise.trial_id is not None

    closes = np.array([b.close for b in bars])
    leak = np.full(len(bars), np.nan)
    leak[:-4] = np.log(closes[4:] / closes[:-4])
    res_leak = evaluate_feature("leaked_future_return", leak, prepared=prep, registry=reg)  # type: ignore[arg-type]
    assert res_leak.passed is True
    assert res_leak.ic_by_horizon["1h"] is not None and res_leak.ic_by_horizon["1h"] > 0.5

    records = reg.records("feature_eval")
    assert [r.model_id for r in records] == ["feature:noise_control", "feature:leaked_future_return"]
    assert records[0].passed_all_gates is False and records[1].passed_all_gates is True
    assert records[1].total_trials_used == 2
    assert "logloss_delta" in records[0].metrics


def test_p1_8_existing_three_features_are_evaluated_leave_one_out(prepared_bars: tuple[list, object]) -> None:  # type: ignore[type-arg]
    bars, _ = prepared_bars
    cfg = FeatureEvalConfig(n_groups=5, k_test_groups=1)
    results = evaluate_existing_features(bars, config=cfg)
    assert set(results) == {"ffd_close_zscore", "ffd_oi_zscore", "oi_funding_basis_zscore"}
    for r in results.values():
        assert r.folds_evaluated >= 3
        assert r.oof_logloss_baseline is not None and r.oof_logloss_candidate is not None
        assert set(r.ic_by_horizon) == {"15m", "1h", "4h", "12h", "24h"}
        assert r.reasons
