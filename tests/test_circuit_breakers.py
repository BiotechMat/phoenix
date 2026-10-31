"""P3.2 tests: every breaker on its own, then store-level behaviour on scripted paths and a calm replay."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.contracts import ClosedTradeRecord, PositionExitReason
from research.replay import ReplayScenario, ScenarioData, run_replay
from risk.circuit_breakers import CircuitBreakerConfig, CircuitBreakers, OperationalStatus
from tests.phase0_helpers import (
    BASE_TS_MS,
    STEP_MS,
    ForcedSignalDerivativesSleeve,
    build_store,
    companion_flat_bars,
    flat_bars,
    flat_history,
    make_snapshot,
    ohlc_path,
)

DAY_NS = 86_400 * 1_000_000_000


def _trade(net: float, reason: PositionExitReason = PositionExitReason.VERTICAL_EXPIRY, sleeve: str = "derivatives_alpha", asset: str = "BTC") -> ClosedTradeRecord:
    return ClosedTradeRecord(
        trade_id="t", position_id="p", asset=asset, sleeve_id=sleeve, side=1, hedged=False, entry_price=100.0, exit_price=100.0,
        entry_bar_index=0, exit_bar_index=1, entry_ts_ns=0, exit_ts_ns=1, size=1.0, notional_usd=100.0, exit_reason=reason,
        gross_pnl_usd=net, funding_pnl_usd=0.0, fee_usd=0.0, net_pnl_usd=net, net_return_bps=net * 100, bars_held=1, is_win=net > 0,
    )


def test_stoploss_guard_locks_after_three_stops_in_48_bars() -> None:
    cb = CircuitBreakers(CircuitBreakerConfig(cooldown_bars=0))
    for bar in (10, 20, 30):
        assert cb.permission("derivatives_alpha", "BTC", bar).allowed
        cb.record_closed_trade(_trade(-1.0, PositionExitReason.STOP_BARRIER), bar)
    p = cb.permission("derivatives_alpha", "BTC", 31)
    assert not p.allowed and "StoplossGuard" in p.reasons[0]
    assert not cb.permission("derivatives_alpha", "BTC", 54).allowed  # locked through bar 30 + 24
    assert cb.permission("derivatives_alpha", "BTC", 55).allowed
    assert cb.permission("derivatives_alpha", "ETH", 31).allowed  # other pairs unaffected

    spread = CircuitBreakers(CircuitBreakerConfig(cooldown_bars=0))
    for bar in (0, 30, 60):  # three stops, but not within 48 bars of each other
        spread.record_closed_trade(_trade(-1.0, PositionExitReason.STOP_BARRIER), bar)
    assert spread.permission("derivatives_alpha", "BTC", 61).allowed


def test_cooldown_blocks_re_entry_for_four_bars() -> None:
    cb = CircuitBreakers()
    cb.record_closed_trade(_trade(+1.0, PositionExitReason.PROFIT_BARRIER), 10)
    assert [cb.permission("derivatives_alpha", "BTC", b).allowed for b in range(10, 16)] == [False, False, False, False, False, True]


def test_low_profit_disables_pair_until_re_enabled() -> None:
    cb = CircuitBreakers(CircuitBreakerConfig(cooldown_bars=0))
    for i in range(19):
        cb.record_closed_trade(_trade(-0.1 if i % 2 else 0.05), i)
    assert cb.permission("derivatives_alpha", "BTC", 100).allowed  # only 19 trades so far
    cb.record_closed_trade(_trade(-0.1), 19)
    p = cb.permission("derivatives_alpha", "BTC", 100)
    assert not p.allowed and "LowProfit" in p.reasons[0]
    cb.re_enable(pair="derivatives_alpha:BTC")
    assert cb.permission("derivatives_alpha", "BTC", 100).allowed


def test_soft_limit_halves_and_hard_limit_disables_a_sleeve() -> None:
    cfg = CircuitBreakerConfig(sleeve_allocation_fraction={"trend": 0.5}, daily_loss_budget=0.5)  # isolate the drawdown limits
    cb = CircuitBreakers(cfg)
    equity = 10_000.0  # trend allocation 5,000: soft at -250, hard at -375
    cb.update_bar(ts_ns=0, equity_usd=equity, sleeve_cum_pnl={"trend": 100.0})
    full = cb.permission("trend", "BTC", 1)
    cb.update_bar(ts_ns=1, equity_usd=equity, sleeve_cum_pnl={"trend": -160.0})  # drawdown 260 from the 100 peak
    half = cb.permission("trend", "BTC", 2)
    assert half.allowed and half.size_multiplier == pytest.approx(full.size_multiplier * 0.5, rel=0.05)
    cb.update_bar(ts_ns=2, equity_usd=equity, sleeve_cum_pnl={"trend": -280.0})  # drawdown 380: hard limit
    hard = cb.permission("trend", "BTC", 3)
    assert not hard.allowed and "hard drawdown" in hard.reasons[0]
    cb.update_bar(ts_ns=3, equity_usd=equity, sleeve_cum_pnl={"trend": -100.0})
    assert not cb.permission("trend", "BTC", 4).allowed  # stays off until re-promoted
    cb.re_enable(sleeve_id="trend")
    assert cb.permission("trend", "BTC", 5).allowed
    assert "trend" in cb.snapshot(5).sleeve_drawdowns


def test_daily_loss_budget_blocks_until_next_utc_day() -> None:
    cb = CircuitBreakers(CircuitBreakerConfig(sleeve_allocation_fraction={"meta_labeller": 1.0}, daily_loss_budget=0.03, hard_limit=0.5, soft_limit=0.4))
    cb.update_bar(ts_ns=10 * DAY_NS, equity_usd=1000.0, sleeve_cum_pnl={"meta_labeller": 0.0})
    cb.update_bar(ts_ns=10 * DAY_NS + 3600 * 10**9, equity_usd=1000.0, sleeve_cum_pnl={"meta_labeller": -31.0})
    p = cb.permission("meta_labeller", "BTC", 1)
    assert not p.allowed and "daily loss budget" in p.reasons[0]
    cb.update_bar(ts_ns=11 * DAY_NS + 60 * 10**9, equity_usd=1000.0, sleeve_cum_pnl={"meta_labeller": -31.0})
    assert cb.permission("meta_labeller", "BTC", 2).allowed


def test_grossman_zhou_multiplier_with_decaying_running_max() -> None:
    cfg = CircuitBreakerConfig(max_portfolio_drawdown=0.20, gz_smoothing_bars=1, running_max_decay_per_day=0.01)
    cb = CircuitBreakers(cfg)
    cb.update_bar(ts_ns=0, equity_usd=1000.0, sleeve_cum_pnl={})
    assert cb.portfolio_multiplier == pytest.approx(1.0)
    cb.update_bar(ts_ns=1, equity_usd=900.0, sleeve_cum_pnl={})  # 10% below the peak, half the allowed 20%
    assert cb.portfolio_multiplier == pytest.approx(0.5)
    cb.update_bar(ts_ns=2, equity_usd=800.0, sleeve_cum_pnl={})  # at the floor
    assert cb.portfolio_multiplier == pytest.approx(0.0)
    assert not cb.permission("trend", "BTC", 3).allowed
    cb.update_bar(ts_ns=2 + 30 * DAY_NS, equity_usd=800.0, sleeve_cum_pnl={})  # the old peak decays by 1%/day
    expected_max = max(800.0, 1000.0 * 0.99**30)
    assert cb.snapshot(None).decayed_running_max == pytest.approx(expected_max)
    assert cb.portfolio_multiplier == pytest.approx((800.0 - 0.8 * expected_max) / (0.2 * expected_max))


def test_operational_lockdown_blocks_entries_and_clears() -> None:
    cb = CircuitBreakers()
    cb.update_bar(ts_ns=0, equity_usd=1000.0, sleeve_cum_pnl={}, operational=OperationalStatus(feed_age_seconds=45.0))
    p = cb.permission("trend", "BTC", 1)
    assert cb.lockdown and not p.allowed and "stale feed" in p.reasons[0]
    assert any("cancel resting orders" in e for e in cb.snapshot(1).events)
    cb.update_bar(ts_ns=1, equity_usd=1000.0, sleeve_cum_pnl={}, operational=OperationalStatus(recent_rejects=6))
    assert "reject storm" in cb.permission("trend", "BTC", 2).reasons[0]
    cb.update_bar(ts_ns=2, equity_usd=1000.0, sleeve_cum_pnl={}, operational=OperationalStatus(position_state_known=False))
    assert "unknown position state" in cb.permission("trend", "BTC", 3).reasons[0]
    cb.update_bar(ts_ns=3, equity_usd=1000.0, sleeve_cum_pnl={}, operational=OperationalStatus())
    assert not cb.lockdown and cb.permission("trend", "BTC", 4).allowed


def _stop_storm_steps(n: int = 40) -> list[tuple[float, float, float, float]]:
    """Flat bars at a falling level with a stop-sized dip every few bars."""
    steps: list[tuple[float, float, float, float]] = []
    level = 100.0
    for i in range(n):
        if i in (1, 7, 13):
            steps.append((level, level, level * 0.997, level * 0.9975))  # dips 30 bps: trips a 20 bps stop
            level *= 0.9975
        else:
            steps.append((level, level, level, level))
    return steps


def test_store_stop_storm_locks_the_pair_and_records_why(tmp_path: Path) -> None:
    steps = _stop_storm_steps()
    store = build_store(
        tmp_path,
        live_bars={"BTC": ohlc_path("BTC", steps), **companion_flat_bars(len(steps))},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
        derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.8, edge_bps=60.0),
    )
    store._breakers = CircuitBreakers()
    reasons_seen: list[str] = []
    for _ in range(len(steps)):
        store.step_simulation_tick()
        reasons_seen.extend(store._blocked_entries)
    stops = [t for t in store.position_manager.closed_trades if t.exit_reason == PositionExitReason.STOP_BARRIER]
    assert len(stops) == 3
    assert any("cooldown" in r for r in reasons_seen)  # re-entry waited after each exit
    assert any("StoplossGuard" in r for r in reasons_seen)  # then the pair locked
    snap = store.sleeve_telemetry["circuit_breakers"]
    assert any(e.startswith("StoplossGuard: derivatives_alpha:BTC") for e in snap["events"])
    # While locked nothing opened: the last trade is the third stop, and any re-entry came after the lock ended.
    assert store.position_manager.closed_trades[-1].exit_reason == PositionExitReason.STOP_BARRIER
    lock_event = next(e for e in snap["events"] if e.startswith("StoplossGuard: derivatives_alpha:BTC"))
    lock_until = int(lock_event.rsplit(" ", 1)[-1])
    reopened = store.position_manager.open_positions.get("derivatives_alpha:BTC")
    assert reopened is None or reopened.entry_bar_index > lock_until
    assert lock_until - stops[-1].exit_bar_index == 24


def test_calm_replay_has_no_false_lockdowns(tmp_path: Path) -> None:
    n = 120
    live = {"BTC": flat_bars("BTC", 0, n, 100.0), **companion_flat_bars(n)}
    data = ScenarioData(history_by_coin={"BTC": flat_history("BTC", 64, 100.0)}, live_by_coin=live, hourly_closes_by_coin={})
    sc = ReplayScenario(
        name="calm_unit",
        start_utc=datetime.fromtimestamp(BASE_TS_MS / 1000, tz=timezone.utc),
        end_utc=datetime.fromtimestamp((BASE_TS_MS + n * STEP_MS) / 1000, tz=timezone.utc),
        warmup_bars=64,
    )
    report = run_replay(sc, data, report_dir=None, circuit_breakers=CircuitBreakers())
    assert report.bars_processed == n
    assert not any("lockdown" in e for e in report.breaker_events)
    assert report.breaker_events == []
