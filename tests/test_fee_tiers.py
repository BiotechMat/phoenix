"""P4.4 tests: fee table parsing, tiers, staking discount, carry hurdle by tier, hedge venue, order slicing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.fee_tiers import best_hedge_venue, fee_schedule_for, parse_user_fees, rates_for_volume, slice_parent_order, tier_index
from sleeves.sleeve_3_funding_carry import FundingCarryConfig, FundingCarrySleeve
from tests.phase0_helpers import companion_flat_bars, flat_bars, flat_history, make_snapshot

FIX = Path(__file__).resolve().parent / "fixtures" / "hyperliquid_user_fees.json"


def _info():  # type: ignore[no-untyped-def]
    return parse_user_fees(json.loads(FIX.read_text(encoding="utf-8")))


def test_parse_recorded_user_fees() -> None:
    info = _info()
    t = info.table
    assert t.base_taker_rate == 0.00045 and t.base_maker_rate == 0.00015
    assert [x.cutoff_usd for x in t.tiers][:3] == [5e6, 25e6, 100e6]
    assert t.tiers[2].maker_rate == 0.00004 and t.tiers[3].maker_rate == 0.0
    assert (0.0001, 0.05) in t.staking_discounts
    assert info.user_maker_rate == 0.00004 and info.volume_14d_usd > 100e6  # this account sits in tier 3
    assert tier_index(t, info.volume_14d_usd) == 3


def test_rates_by_volume_and_staking_discount() -> None:
    t = _info().table
    assert rates_for_volume(t, 0.0) == (0.00045, 0.00015)
    assert rates_for_volume(t, 30e6) == (0.00035, 0.00008)
    assert rates_for_volume(t, 150e6, staking_discount=0.10) == pytest.approx((0.00027, 0.000036))
    assert rates_for_volume(t, 600e6, staking_discount=0.10)[1] == 0.0  # zero maker stays zero
    sched = fee_schedule_for(t, 150e6)
    assert sched.maker_bps == pytest.approx(0.4) and sched.taker_bps == pytest.approx(3.0)


def test_carry_trade_rejected_at_tier_0_and_accepted_at_tier_3() -> None:
    t = _info().table
    sleeve_cfg = FundingCarryConfig(min_positive_funding_8h=0.0001)
    funding_8h = 0.00015  # 1.5 bps per 8 hours
    tier0 = FundingCarrySleeve(sleeve_cfg).evaluate_opportunity("BTC", funding_rate_8h=funding_8h, fee_schedule=fee_schedule_for(t, 0.0))
    tier3 = FundingCarrySleeve(sleeve_cfg).evaluate_opportunity("BTC", funding_rate_8h=funding_8h, fee_schedule=fee_schedule_for(t, 150e6))
    assert tier0.is_actionable is False and tier0.expected_net_carry_bps < 8.0  # 13.5 bps carry minus 7.5 bps of costs
    assert tier3.is_actionable is True and tier3.roundtrip_cost_bps < tier0.roundtrip_cost_bps


def test_best_hedge_venue_nets_out_fees() -> None:
    choice = best_hedge_venue({"HlPerp": 0.0003, "BinPerp": 0.0006, "BybitPerp": 0.00055}, {"HlPerp": 0.0004, "BinPerp": 0.0015, "BybitPerp": 0.0007})
    assert choice is not None and choice.venue == "BybitPerp"  # Binance pays more funding but costs more to trade
    assert choice.net_funding_per_day == pytest.approx(0.00055 - 0.0007 / 3.0)
    assert best_hedge_venue({}, {}) is None


def test_parent_order_slicing() -> None:
    assert [s.notional_usd for s in slice_parent_order(1_000.0, 1_000_000.0)] == [1_000.0]  # under 2% of a minute's volume
    slices = slice_parent_order(50_000.0, 1_000_000.0)
    assert len(slices) == 3 and sum(s.notional_usd for s in slices) == pytest.approx(50_000.0)
    assert [s.passive for s in slices] == [True, True, False]
    assert slice_parent_order(0.0, 1.0) == []


def test_store_derives_fee_schedule_from_table_and_volume(tmp_path: Path) -> None:
    from core.data_ingestion import ScriptedBarFeed
    from ui.api_server import ControlCenterStateStore

    feed = ScriptedBarFeed(bars_by_coin={"BTC": flat_bars("BTC", 0, 1, 100.0), **companion_flat_bars(1)}, snapshots=[make_snapshot()], history_by_coin={"BTC": flat_history("BTC", 64, 100.0)})
    store = ControlCenterStateStore(ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False, fee_table=_info().table, account_volume_14d_usd=30e6)
    tele = store.fee_telemetry
    assert tele["tier"] == 2 and tele["maker_bps"] == pytest.approx(0.8) and tele["taker_bps"] == pytest.approx(3.5)
    assert store.position_manager.fee_schedule.maker_bps == pytest.approx(0.8)  # fills are charged at the tier rate
