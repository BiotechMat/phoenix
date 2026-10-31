"""P2.1 tests: deviation rule, state machine, settlement accounting, inversion bound, normalisation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from data.backfill_storage import FundingSettlement, load_settlements, normalize_funding_to_hourly, save_settlements
from research.carry_eval import _round_trips, evaluate_asset, inversion_bound, simulate
from research.trial_registry import TrialRegistry
from sleeves.sleeve_3_funding_carry import CarryV2Config, carry_v2_decide, compute_deviation_rho

H = 3_600_000


def test_rho_uses_hyperliquid_premium_frequency() -> None:
    cfg = CarryV2Config()
    assert cfg.kappa == 1095.0 and cfg.interest_annual == pytest.approx(0.1095)
    assert compute_deviation_rho(0.0001, cfg) == pytest.approx(0.0)  # premium equal to the 8h interest: fair
    assert compute_deviation_rho(0.0009, cfg) == pytest.approx(1095 * 0.0008)
    assert cfg.upper_band == pytest.approx(1095 * math.log(1.00075))


def test_state_machine_entry_exit_and_crowding() -> None:
    cfg = CarryV2Config(roundtrip_cost=0.00075)
    flat_in_band = carry_v2_decide(in_position=False, premium=0.0005, funding_per_hour=0.00002, trailing_funding_per_hour=0.00002, config=cfg)
    assert flat_in_band.action == "FLAT" and flat_in_band.size == 0.0  # rho 0.4380 < band 0.8211
    enter = carry_v2_decide(in_position=False, premium=0.0012, funding_per_hour=0.00002, trailing_funding_per_hour=0.00002, config=cfg)
    assert enter.action == "ENTER" and enter.size == 1.0
    no_entry_neg_trailing = carry_v2_decide(in_position=False, premium=0.0012, funding_per_hour=0.00002, trailing_funding_per_hour=-0.000001, config=cfg)
    assert no_entry_neg_trailing.action == "FLAT"
    crowded = carry_v2_decide(in_position=False, premium=0.0012, funding_per_hour=0.00003, trailing_funding_per_hour=0.00002, config=cfg)
    assert crowded.size == 0.5  # 0.00003 * 8760 = 26% annualised > 25%
    assert carry_v2_decide(in_position=True, premium=0.0001, funding_per_hour=0.00001, trailing_funding_per_hour=0.00001, config=cfg).action == "EXIT"
    assert carry_v2_decide(in_position=True, premium=0.0005, funding_per_hour=0.00001, trailing_funding_per_hour=-0.000001, config=cfg).action == "EXIT"
    assert carry_v2_decide(in_position=True, premium=0.0005, funding_per_hour=-0.0005, trailing_funding_per_hour=0.00001, config=cfg).action == "EXIT"
    assert carry_v2_decide(in_position=True, premium=0.0005, funding_per_hour=0.00001, trailing_funding_per_hour=0.00001, config=cfg).action == "HOLD"


def _settlements(premiums: list[float], rate: float = 0.00002) -> list[FundingSettlement]:
    return [FundingSettlement(t_ms=1_700_000_000_000 + i * H, rate_per_hour=rate, premium=p, interval_hours=1) for i, p in enumerate(premiums)]


def test_simulation_accounting_funding_basis_and_costs() -> None:
    # Enter at step 1 (premium 0.0012 above the band), hold, exit when the premium returns to fair at step 4.
    st = _settlements([0.0005, 0.0012, 0.0008, 0.0006, 0.0001, 0.0001])
    sim = simulate(st, "v2")
    assert list(sim["size"]) == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert list(sim["held"]) == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert sum(sim["funding"]) == pytest.approx(3 * 0.00002)  # three settlements while short
    assert sum(sim["basis"]) == pytest.approx(0.0012 - 0.0001)  # premium convergence from entry to exit
    rt = 7.5e-4  # perp 2 x (1.5 + 0.5) bps + spot 2 x (1.5 + 0.25) bps
    assert sum(sim["cost"]) == pytest.approx(rt)
    assert sum(sim["net"]) == pytest.approx(3 * 0.00002 + 0.0011 - rt)
    assert _round_trips(sim["size"]) == 1


def test_inversion_bound_is_the_worst_eleven_day_funding_loss() -> None:
    rates = [0.00001] * 400 + [-0.00005] * 300 + [0.00001] * 400
    st = [FundingSettlement(t_ms=i * H, rate_per_hour=r, premium=0.0001, interval_hours=1) for i, r in enumerate(rates)]
    # 11 days = 264 hours: the worst window lies fully inside the negative stretch (265 settlements inclusive).
    assert inversion_bound(st, days=11) == pytest.approx(265 * 0.00005, rel=0.01)


def test_eight_hour_settlements_are_normalised_to_hourly() -> None:
    raw = [(0, 0.0001), (8 * H, 0.0001), (16 * H, 0.0001), (17 * H, 0.0000125), (18 * H, 0.0000125)]
    hourly = normalize_funding_to_hourly(raw)
    assert [r for _, r in hourly] == pytest.approx([0.0000125] * 5)


def test_settlements_round_trip_through_parquet_and_registry(tmp_path: Path) -> None:
    st = _settlements([0.0005, 0.0012, 0.0008, 0.0006, 0.0001, 0.0001] * 200)
    path = tmp_path / "btc_funding.parquet"
    assert save_settlements(st, path) == len(st)
    assert load_settlements(path) == st
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    res = evaluate_asset("BTC", st, "v2", reg)
    assert reg.count("carry") == 1 and res.trial_id
    assert res.round_trips == 200 and res.aug_2024_drawdown is None  # synthetic window lies outside August 2024
