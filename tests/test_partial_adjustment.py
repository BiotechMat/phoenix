"""P3.1 tests: adjustment rate, grid, impact, uncertainty-scaled Kelly, transfer coefficient, store wiring."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from risk.partial_adjustment import (
    PartialAdjuster,
    PartialAdjusterConfig,
    adjustment_rate,
    discretise,
    square_root_impact,
    transfer_coefficient,
    uncertainty_kelly_gamma,
)
from tests.phase0_helpers import ForcedSignalDerivativesSleeve, build_store, companion_flat_bars, flat_bars, flat_history, make_snapshot


def test_adjustment_rate_and_grid() -> None:
    assert adjustment_rate(1.0, 1.0) == pytest.approx(1.0 - math.exp(-1.0))
    assert adjustment_rate(1.0, 0.0) == 1.0
    assert adjustment_rate(4.0, 4.0 / math.log(2.0)) == pytest.approx(0.5)  # one half-life
    assert discretise(0.312) == pytest.approx(0.30)
    assert discretise(0.326) == pytest.approx(0.35)  # nearest, not toward zero
    assert discretise(-0.074) == pytest.approx(-0.05)
    assert discretise(0.024) == 0.0
    assert discretise(0.33, step=0.0) == 0.33


def test_partial_adjuster_path_converges_and_flattens_on_cleared_or_flipped_aims() -> None:
    adj = PartialAdjuster(PartialAdjusterConfig(half_life_bars={"trend": 1.0 / math.log(2.0)}, step=0.05))  # lambda = 0.5
    path = adj.path("trend", [0.4] * 6 + [0.0, -0.4])
    assert path[:4] == pytest.approx([0.2, 0.3, 0.35, 0.4])  # halves the gap each bar, on the grid
    assert path[5] == pytest.approx(0.4)
    assert path[6] == 0.0  # aim cleared: flat at once
    assert path[7] == pytest.approx(-0.2)  # flipped: starts again from zero
    assert PartialAdjuster().path("other", [0.4, 0.4]) == pytest.approx([0.4, 0.4])  # no half-life: jump to aim


def test_square_root_impact_and_uncertainty_kelly() -> None:
    assert square_root_impact(0.04, 1_000_000.0, 1_000_000_000.0) == pytest.approx(0.5 * 0.04 * math.sqrt(0.001))
    assert square_root_impact(0.04, 1_000_000.0, 0.0) == 0.0  # unknown volume: no deduction rather than a guess
    assert uncertainty_kelly_gamma(0.02, 0.005) == 0.5
    assert uncertainty_kelly_gamma(0.01, 0.01) == pytest.approx(0.25)
    assert uncertainty_kelly_gamma(-0.01, 0.01) == 0.0


def test_transfer_coefficient() -> None:
    assert transfer_coefficient([1.0, 0.5, -1.0], [0.2, 0.1, -0.2]) == pytest.approx(1.0)
    assert transfer_coefficient([1.0, 0.5, -1.0], [-0.2, -0.1, 0.2]) == pytest.approx(-1.0)
    assert transfer_coefficient([0.0, 0.0, 0.0], [0.1, 0.0, 0.0]) is None


def test_store_reports_transfer_coefficient_and_deducts_impact(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0), **companion_flat_bars(1)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
        derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.8, edge_bps=60.0),
    )
    store.step_simulation_tick()
    assert store.transfer_coefficient is not None and store.transfer_coefficient > 0.9  # the only signal is executed as given
    impact = store._impact_bps("BTC", 1 / 3, 0.002)
    expected = square_root_impact(0.002 * math.sqrt(96.0), (1 / 3) * 5000.0 * 0.5, 1_000_000.0) * 1e4
    assert impact == pytest.approx(expected) and impact > 0.0


def test_payoff_aware_kelly_breaks_even_at_one_over_one_plus_b() -> None:
    from core.contracts import AssetAllocationInput, FeeSchedule, compute_fractional_kelly, compute_signed_target_weights

    assert compute_fractional_kelly(0.6) == pytest.approx(0.5 * 0.2)  # even odds unchanged
    assert compute_fractional_kelly(0.4) == 0.0
    b = 0.008 / 0.003  # 80 bp profit barrier against a 30 bp stop
    assert compute_fractional_kelly(1.0 / (1.0 + b), payoff_ratio=b) == pytest.approx(0.0, abs=1e-12)
    assert compute_fractional_kelly(0.4, payoff_ratio=b) == pytest.approx(0.5 * (0.4 - 0.6 / b))
    with pytest.raises(ValueError):
        compute_fractional_kelly(0.4, payoff_ratio=0.0)
    fee = FeeSchedule.hyperliquid()
    even = compute_signed_target_weights([AssetAllocationInput(asset_id="BTC", y_primary=1, w_hrp=1 / 3, p_meta=0.4, expected_gross_return_bps=12.0, fee_schedule=fee)], regime_multiplier=1.0)
    payoff = compute_signed_target_weights([AssetAllocationInput(asset_id="BTC", y_primary=1, w_hrp=1 / 3, p_meta=0.4, expected_gross_return_bps=12.0, fee_schedule=fee, payoff_ratio=b)], regime_multiplier=1.0)
    assert even.weights["BTC"] == 0.0 and payoff.weights["BTC"] == pytest.approx((1 / 3) * 0.5 * (0.4 - 0.6 / b))
