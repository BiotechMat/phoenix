"""P5.1 tests: pre-registered kill thresholds, shadow book statistics and kills, store routing to the shadow book."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from core.contracts import ClosedTradeRecord, FeeSchedule, PositionExitReason
from research.validation.shadow import PreRegisteredThresholds, ShadowBook, register_thresholds
from sleeves.sleeve_2_trend import TREND_SLEEVE_ID, TrendSleeve, TrendSleeveConfig
from tests.phase0_helpers import ForcedSignalDerivativesSleeve, barrier_path_steps, companion_flat_bars, flat_bars, flat_history, make_snapshot, ohlc_path
from tests.test_trend import _trend_hourly


def _trade(net: float, i: int) -> ClosedTradeRecord:
    return ClosedTradeRecord(
        trade_id=f"t{i}", position_id=f"p{i}", asset="BTC", sleeve_id="trend", side=1, hedged=False, entry_price=100.0, exit_price=100.0,
        entry_bar_index=i, exit_bar_index=i + 1, entry_ts_ns=0, exit_ts_ns=1, size=10.0, notional_usd=1000.0, exit_reason=PositionExitReason.TARGET_CLEARED,
        gross_pnl_usd=net + 0.3, funding_pnl_usd=0.0, fee_usd=0.3, net_pnl_usd=net, net_return_bps=net * 10, bars_held=1, is_win=net > 0,
    )


def test_thresholds_from_cpcv_paths() -> None:
    t = np.arange(500, dtype=np.float64)
    paths = np.vstack([0.0002 + 0.001 * np.sin(t / (3 + k)) for k in range(9)])
    th = register_thresholds("trend", paths, periods_per_year=8760.0)
    sharpes = [float(np.mean(p) / np.std(p, ddof=1) * math.sqrt(8760.0)) for p in paths]
    assert th.median_path_sharpe == pytest.approx(float(np.median(sharpes)))
    assert th.min_live_sharpe == pytest.approx(0.8 * th.median_path_sharpe)
    assert th.n_paths == 9 and th.max_drawdown > 0.0


def test_shadow_book_kills_on_drawdown_and_reconciles_costs() -> None:
    th = PreRegisteredThresholds(sleeve_id="trend", median_path_sharpe=1.0, min_live_sharpe=0.8, max_drawdown=0.01, n_paths=9)
    book = ShadowBook("trend", th, min_trades=30, capital_usd=1000.0)
    for i in range(25):
        rec = book.record(_trade(-1.0, i), FeeSchedule.hyperliquid(), exit_markout_1s_bps=-2.0)
    st = book.status()
    assert st.killed and "drawdown" in (st.kill_reason or "") and st.state == "KILLED" and not st.eligible_for_promotion
    assert rec.modelled_cost_bps == pytest.approx(4.0)  # 2 x (1.5 maker + 0.5 slippage)
    assert rec.realised_cost_bps == pytest.approx(3.0 + 2.0)  # 3 bps of fees + 2 bps of adverse markout
    assert st.mean_cost_gap_bps == pytest.approx(1.0)


def test_shadow_book_becomes_eligible_after_min_trades_without_a_kill() -> None:
    th = PreRegisteredThresholds(sleeve_id="trend", median_path_sharpe=1.0, min_live_sharpe=0.8, max_drawdown=0.5, n_paths=9)
    book = ShadowBook("trend", th, min_trades=40, capital_usd=1000.0)
    for i in range(40):
        book.record(_trade(1.0 if i % 3 else -0.5, i), FeeSchedule.hyperliquid())
    st = book.status()
    assert st.trades == 40 and not st.killed and st.eligible_for_promotion and st.state == "ELIGIBLE"
    assert ShadowBook("x", None).status().eligible_for_promotion is False  # no pre-registered thresholds, no promotion


def test_unvalidated_derivatives_sleeve_trades_in_the_shadow_book(tmp_path: Path) -> None:
    from core.data_ingestion import ScriptedBarFeed
    from sleeves.sleeve_1_derivatives_alpha import DerivativesAlphaConfig
    from ui.api_server import ControlCenterStateStore

    class _Unvalidated(ForcedSignalDerivativesSleeve):
        def __init__(self) -> None:
            super().__init__(asset="BTC", direction=1, p_meta=0.8, edge_bps=60.0)
            self.config = DerivativesAlphaConfig(allocate=False)

    steps = barrier_path_steps()
    feed = ScriptedBarFeed(bars_by_coin={"BTC": ohlc_path("BTC", steps), **companion_flat_bars(len(steps))}, snapshots=[make_snapshot()], history_by_coin={"BTC": flat_history("BTC", 64, 100.0)})
    store = ControlCenterStateStore(ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False, derivatives_sleeve=_Unvalidated())
    for _ in range(len(steps)):
        store.step_simulation_tick()
    assert store.position_manager.closed_trades == [] and store.ledger_summary["total_trades"] == 0  # nothing live
    shadow_closed = store.shadow_positions.closed_trades
    assert [t.exit_reason for t in shadow_closed] == [PositionExitReason.PROFIT_BARRIER, PositionExitReason.STOP_BARRIER, PositionExitReason.VERTICAL_EXPIRY]
    tele = store.shadow_telemetry
    assert tele["sleeves"] == ["derivatives_alpha", "meta_labeller_challenger"]  # the P2.3 challenger also runs in shadow
    assert tele["status"]["derivatives_alpha"]["trades"] == 3 and tele["status"]["derivatives_alpha"]["state"] == "SHADOW"


def test_enabled_trend_sleeve_defaults_to_shadow(tmp_path: Path) -> None:
    from core.data_ingestion import ScriptedBarFeed
    from ui.api_server import ControlCenterStateStore

    hourly = _trend_hourly()
    last_px = {c: v[-1][1] for c, v in hourly.items()}
    feed = ScriptedBarFeed(
        bars_by_coin={c: flat_bars(c, 0, 1, last_px[c]) for c in ("BTC", "ETH", "SOL")}, snapshots=[make_snapshot(prices=last_px)],
        history_by_coin={"BTC": flat_history("BTC", 64, last_px["BTC"])}, hourly_closes_by_coin=hourly,
    )
    store = ControlCenterStateStore(ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False, trend_sleeve=TrendSleeve(TrendSleeveConfig(enabled=True)))
    store.step_simulation_tick()
    assert f"{TREND_SLEEVE_ID}:BTC" not in store.position_manager.open_positions
    assert f"{TREND_SLEEVE_ID}:BTC" in store.shadow_positions.open_positions
    assert TREND_SLEEVE_ID in store.shadow_telemetry["sleeves"]
