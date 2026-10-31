"""P1.6 tests: trial registry counts, DSR deflation on repeated trials, and the MinBTL refusal."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from core.contracts import CPCVValidationResult, FeeSchedule
from research.trial_registry import (
    BARS_PER_YEAR_15M,
    TrialRecord,
    TrialRegistry,
    compute_min_backtest_length_years,
    expected_max_sharpe_of_null_trials,
    passes_min_backtest_length,
    run_registered_validation,
)
from research.validation.cpcv_gatekeeper import ChampionChallengerGatekeeper, CombinatorialPurgedCV, compute_dsr


def _deterministic_returns(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Noise-like test returns with a modest per-bar Sharpe so the DSR is neither 0 nor saturated at 1."""
    rng = np.random.default_rng(7)  # test fixture only; never a live path
    gross = 0.0001 + 0.0012 * rng.standard_normal(n)
    positions = np.ones(n, dtype=np.float64)
    return gross, positions


def test_registry_appends_and_counts_per_family(tmp_path: Path) -> None:
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    assert reg.count("carry") == 0 and reg.next_total_trials("carry") == 1
    result = CPCVValidationResult(model_id="m1", purge_bars=16, psr_vs_zero=0.9, psr_vs_benchmark=0.9, dsr=0.8, pbo=0.2,
                                  expected_net_pnl_bps=20.0, sharpe_net=1.1, skewness=0.0, kurtosis=3.0, total_trials=1)
    rec = reg.append(TrialRecord.from_result("carry", result, features=("funding", "basis"), parameters={"k": 8760}, n_bars=600, seed=7))
    assert rec.sleeve_family == "carry" and rec.metrics["dsr"] == 0.8 and rec.features == ("funding", "basis")
    reg.append(TrialRecord.from_result("trend", result))
    assert reg.count("carry") == 1 and reg.count("trend") == 1 and reg.next_total_trials("carry") == 2
    assert len(reg.records()) == 2
    assert reg.records("carry")[0].parameters == {"k": 8760}


def test_same_configuration_twice_increments_count_and_lowers_dsr(tmp_path: Path) -> None:
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    gross, positions = _deterministic_returns()
    cpcv = CombinatorialPurgedCV(vertical_barrier_bars=16, n_groups=6, k_test_groups=2, embargo_pct=0.01)
    res1, rec1 = run_registered_validation(reg, "carry", "cfg_a", gross, positions, cpcv=cpcv, fee_schedule=FeeSchedule.hyperliquid(), parameters={"x": 1})
    res2, rec2 = run_registered_validation(reg, "carry", "cfg_a", gross, positions, cpcv=cpcv, fee_schedule=FeeSchedule.hyperliquid(), parameters={"x": 1})
    assert rec1.total_trials_used == 1 and rec2.total_trials_used == 2
    assert reg.count("carry") == 2
    assert res1.psr_vs_zero == pytest.approx(res2.psr_vs_zero)  # same data, same PSR
    assert 0.0 < res1.dsr < 1.0
    assert res2.dsr < res1.dsr  # second trial is deflated for selection
    # Direct check of the deflation mechanism on the same return vector.
    rets = gross * positions
    assert compute_dsr(rets, total_trials=2) < compute_dsr(rets, total_trials=1)


def test_min_backtest_length_grows_with_trials_and_shrinks_with_target_sharpe() -> None:
    assert expected_max_sharpe_of_null_trials(1) == 0.0
    assert expected_max_sharpe_of_null_trials(10) > expected_max_sharpe_of_null_trials(2) > 0.0
    assert compute_min_backtest_length_years(1, 1.0) == 0.0
    y10 = compute_min_backtest_length_years(10, 1.0)
    y100 = compute_min_backtest_length_years(100, 1.0)
    assert 0.0 < y10 < y100
    assert compute_min_backtest_length_years(10, 2.0) == pytest.approx(y10 / 4.0)
    assert math.isinf(compute_min_backtest_length_years(10, 0.0))
    assert y10 == pytest.approx(2.48, abs=0.05)  # ten trials at SR* = 1 need about two and a half years
    ok, years, need = passes_min_backtest_length(int(BARS_PER_YEAR_15M * 3), 10, 1.0)
    assert ok and years == pytest.approx(3.0) and need == pytest.approx(y10)
    ok_two, _, _ = passes_min_backtest_length(int(BARS_PER_YEAR_15M * 2), 10, 1.0)
    assert not ok_two
    ok_short, _, _ = passes_min_backtest_length(int(BARS_PER_YEAR_15M * 0.1), 100, 1.0)
    assert not ok_short


def test_gatekeeper_refuses_promotion_when_sample_is_shorter_than_min_btl() -> None:
    passing = CPCVValidationResult(model_id="challenger_x", purge_bars=16, psr_vs_zero=0.99, psr_vs_benchmark=0.98, dsr=0.97, pbo=0.1,
                                   expected_net_pnl_bps=30.0, sharpe_net=2.0, skewness=0.0, kurtosis=3.0, total_trials=200)
    assert passing.passed_all_gates
    gate_short = ChampionChallengerGatekeeper(asset_id="BTC")
    state_short = gate_short.evaluate_challenger(passing, sample_bars=int(BARS_PER_YEAR_15M * 0.25), target_sharpe_annual=1.0)
    assert state_short.champion_model_id is None
    assert "MinBTL" in state_short.last_promotion_reason
    gate_long = ChampionChallengerGatekeeper(asset_id="BTC")
    state_long = gate_long.evaluate_challenger(passing, sample_bars=int(BARS_PER_YEAR_15M * 10), target_sharpe_annual=1.0)
    assert state_long.champion_model_id == "challenger_x"
    # Without a sample length the behaviour is unchanged.
    assert ChampionChallengerGatekeeper(asset_id="BTC").evaluate_challenger(passing).champion_model_id == "challenger_x"
