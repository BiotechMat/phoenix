"""P3.4 tests: fragility conditions, ADL rank, collateral haircut, liquidation stress, store wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.market_feeds import MacroInputs, ScriptedMacroInputsFeed, StablecoinQuote, compute_stablecoin_marks
from risk.cascade_controls import (
    CascadeConfig,
    CascadeController,
    StressPosition,
    adl_rank_estimate,
    collateral_mark,
    liquidation_stress_loss,
)
from tests.phase0_helpers import ForcedSignalDerivativesSleeve, companion_flat_bars, flat_bars, flat_history, make_bar, make_snapshot

H_NS = 3_600 * 1_000_000_000
BAR_NS = 900 * 1_000_000_000


def _feed_calm_then(ctrl: CascadeController, *, compress: bool, oi_spike: bool, funding_1h: float, depth_drop: bool) -> int:
    ts = 0
    for i in range(400):
        ts = i * BAR_NS
        wiggle = 0.3 * ((i % 7) - 3) / 3.0
        imbalance = wiggle * (0.05 if (compress and i >= 340) else 1.0)
        buy, sell = 1000.0 * (1 + imbalance), 1000.0 * (1 - imbalance)
        oi = 1000.0 + (i % 11) + (200.0 if (oi_spike and i == 399) else 0.0)
        depth = 1_000_000.0 * (0.3 if (depth_drop and i >= 396) else 1.0)
        ctrl.update("BTC", ts_ns=ts, taker_buy_usd=buy, taker_sell_usd=sell, open_interest=oi, funding_rate_1h=funding_1h, depth_10bps_usd=depth)
    return ts


def test_each_fragility_condition_and_the_two_condition_trigger() -> None:
    calm = CascadeController()
    ts = _feed_calm_then(calm, compress=False, oi_spike=False, funding_1h=0.00001, depth_drop=False)
    a = calm.assess("BTC", ts)
    assert a.conditions_met == [] and not a.fragile and a.quote_action == "normal" and a.directional_multiplier == 1.0
    assert a.oi_percentile is not None and 0.0 < a.oi_percentile < 0.95

    one = CascadeController()
    ts = _feed_calm_then(one, compress=False, oi_spike=False, funding_1h=0.00004, depth_drop=False)
    b = one.assess("BTC", ts)
    assert b.conditions_met == ["funding_above_25pct"] and not b.fragile and b.quote_action == "widen"

    two = CascadeController()
    ts = _feed_calm_then(two, compress=True, oi_spike=True, funding_1h=0.00004, depth_drop=True)
    c = two.assess("BTC", ts)
    assert set(c.conditions_met) == {"taker_flow_variance_compressed", "open_interest_above_p90", "funding_above_25pct", "depth_down_50pct_1h"}
    assert c.fragile and c.directional_multiplier == 0.5 and c.block_carry_entries and c.quote_action == "pause"


def test_missing_inputs_are_flagged_not_counted() -> None:
    ctrl = CascadeController()
    ctrl.update("ETH", ts_ns=0, taker_buy_usd=None, taker_sell_usd=None, open_interest=None, funding_rate_1h=0.001, depth_10bps_usd=None)
    a = ctrl.assess("ETH", 0)
    assert set(a.conditions_unavailable) == {"taker_flow_variance", "open_interest_history", "depth_history"}
    assert a.conditions_met == ["funding_above_25pct"] and not a.fragile


def test_flat_open_interest_is_not_at_its_highs() -> None:
    ctrl = CascadeController()
    for i in range(50):
        ctrl.update("SOL", ts_ns=i, taker_buy_usd=None, taker_sell_usd=None, open_interest=500.0, funding_rate_1h=None, depth_10bps_usd=None)
    assert ctrl.assess("SOL", 50).oi_percentile == pytest.approx(0.5)


def test_adl_collateral_and_stress() -> None:
    assert adl_rank_estimate(110.0, 100.0, 5000.0, 10000.0) == pytest.approx(0.55)
    agree = collateral_mark([0.9998, 1.0001, 0.9999])
    assert agree is not None and agree.haircut == 0.0 and agree.mark == pytest.approx(0.9999)
    split = collateral_mark([0.98, 1.0, 1.0])
    assert split is not None and split.haircut == 0.10 and split.max_disagreement == pytest.approx(0.02)
    assert collateral_mark([]) is None
    loss = liquidation_stress_loss([StressPosition(asset="BTC", notional_usd=100_000.0, depth_10bps_usd=1_000_000.0)])
    d = 100_000.0  # depth after a 90% loss
    expected = sum((100_000.0 * 0.8**k * 0.2) ** 2 * 0.0010 / d / 2.0 for k in range(5))
    assert loss["BTC"] == pytest.approx(expected)


def test_store_halves_directional_entries_and_blocks_carry_when_fragile(tmp_path: Path) -> None:
    from core.data_ingestion import ScriptedBarFeed
    from ui.api_server import ControlCenterStateStore

    ctrl = CascadeController(CascadeConfig(fragile_conditions=1))  # funding alone suffices in this test
    funding = 0.000125  # 109% annualised: carry is actionable and the asset is "fragile" under the 1-condition rule
    feed = ScriptedBarFeed(
        bars_by_coin={"BTC": flat_bars("BTC", 0, 1, 100.0, funding_1h=funding), **companion_flat_bars(1)},
        snapshots=[make_snapshot(funding_1h={"BTC": funding})],
        history_by_coin={"BTC": flat_history("BTC", 64, 100.0)},
    )
    store = ControlCenterStateStore(
        ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False,
        derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.8, edge_bps=60.0), cascade_controller=ctrl,
    )
    store.step_simulation_tick()
    deriv_target = store.latest_targets["derivatives_alpha:BTC"].weight
    pos = store.position_manager.open_positions["derivatives_alpha:BTC"]
    assert pos.notional_usd == pytest.approx(0.5 * deriv_target * 5000.0)
    assert "funding_carry:BTC" not in store.position_manager.open_positions
    assert any("cascade fragility" in r for r in store._blocked_entries)
    tele = store.sleeve_telemetry["cascade"]
    assert tele["fragility"]["BTC"]["fragile"] is True and tele["collateral_haircut"] == 0.0


def test_store_applies_collateral_haircut_when_usdc_venues_disagree(tmp_path: Path) -> None:
    from core.data_ingestion import ScriptedBarFeed
    from ui.api_server import ControlCenterStateStore

    quotes = [StablecoinQuote(symbol="USDC", venue=v, price=p) for v, p in (("kraken", 0.97), ("bitstamp", 1.0), ("gemini", 1.0))]
    quotes += [StablecoinQuote(symbol="USDT", venue=v, price=1.0) for v in ("coinbase", "kraken")]
    inputs = MacroInputs(stablecoin_depeg_bps=0.0, stablecoin_marks=compute_stablecoin_marks(quotes))
    feed = ScriptedBarFeed(bars_by_coin={"BTC": flat_bars("BTC", 0, 1, 100.0), **companion_flat_bars(1)}, snapshots=[make_snapshot()], history_by_coin={"BTC": flat_history("BTC", 64, 100.0)})
    store = ControlCenterStateStore(
        ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False,
        derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.8, edge_bps=60.0), macro_feed=ScriptedMacroInputsFeed(inputs),
    )
    store.step_simulation_tick()
    assert store.sleeve_telemetry["cascade"]["collateral_haircut"] == 0.10
    target = store.latest_targets["derivatives_alpha:BTC"].weight
    assert store.position_manager.open_positions["derivatives_alpha:BTC"].notional_usd == pytest.approx(target * 5000.0 * 0.9)
    _ = make_bar
