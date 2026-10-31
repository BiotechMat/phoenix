"""Phase 0 acceptance tests (docs/REINFORCEMENT_PLAN.md, tasks P0.1 to P0.7 and the Phase 0 gate).

All tests run offline on scripted bars. No random generator is used anywhere in these fixtures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np
import pytest

from core.contracts import (
    DataAvailability,
    DollarBar,
    ExecutionRegimeMode,
    FeeSchedule,
    MetaPredictionOutput,
    PositionExitReason,
    TripleBarrierConfig,
)
from core.data_ingestion import ScriptedBarFeed, candle_to_dollar_bar, funding_rate_for_time
from core.execution.position_manager import BarMarks, PositionManager, SleeveTarget, position_key
from core.models import EWMAVolatilityEstimator, compute_ewma_volatility
from risk.portfolio_allocator import HRPWeightScheduler, RollingHourlyReturnMatrix
from tests.phase0_helpers import (
    ASSETS,
    BASE_TS_MS,
    DEFAULT_PRICES,
    STEP_MS,
    ForcedSignalDerivativesSleeve,
    barrier_path_steps,
    build_store,
    companion_flat_bars,
    flat_bars,
    flat_history,
    hourly_closes_trending,
    ledger_core_fields,
    make_bar,
    make_snapshot,
    ohlc_path,
    sine_history,
)
from ui.api_server import (
    MIN_TRAIN_LABELLED_BARS,
    RETRAIN_INTERVAL_NS,
    SLEEVE_DERIVATIVES,
    SLEEVE_FUNDING_CARRY,
    ControlCenterStateStore,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# P0.2: no probability floor, no direction defaults
# =============================================================================


def _meta(p: float, direction: int, gate: bool = True) -> MetaPredictionOutput:
    return MetaPredictionOutput(
        instrument_id="BTCUSDT-PERP",
        ts_close_ns=1,
        primary_direction=1 if direction > 0 else (-1 if direction < 0 else 0),
        p_meta=p,
        mahalanobis_di=1.0,
        di_threshold=3.5,
        epistemic_gate_passed=gate,
        svd_factors=(0.1, 0.2, 0.3),
        liquidation_exhaustion_flag=0,
        expected_net_edge_bps=0.0,
    )


def test_p0_2_coin_flip_probability_or_zero_direction_yields_zero_weight(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    mode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
    key = position_key("meta_labeller", "BTC")

    targets_half, _, _ = store._build_targets(_meta(0.50, 1), store.sigma_bar, 1.0, mode)
    assert targets_half[key].weight == 0.0

    targets_zero_dir, _, _ = store._build_targets(_meta(0.80, 0), store.sigma_bar, 1.0, mode)
    assert targets_zero_dir[key].weight == 0.0

    targets_gate_closed, _, _ = store._build_targets(_meta(0.80, 1, gate=False), store.sigma_bar, 1.0, mode)
    assert targets_gate_closed[key].weight == 0.0

    # Sanity: a real edge with an open gate does produce a signed weight, so the zeros above are not vacuous.
    targets_real, alloc, _ = store._build_targets(_meta(0.80, -1), store.sigma_bar, 1.0, mode)
    assert targets_real[key].weight < 0.0
    assert alloc.weights["BTC"] < 0.0


def test_p0_2_tick_with_no_signal_is_hold_cash(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 3, 100.0), **companion_flat_bars(3)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    assert store.ml_worker is None  # no honest model: p_meta = 0.5 and direction 0
    for _ in range(3):
        res = store.step_simulation_tick()
        assert res.bar_processed is True
        assert res.weights == {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
        assert res.gross_leverage == 0.0
        assert res.forensic_record is not None
        assert res.forensic_record["action"] == "HOLD_CASH"
        assert res.forensic_record["side"] == "FLAT"
    assert store.position_manager.open_positions == {}


# =============================================================================
# P0.1: sleeves wired into the tick
# =============================================================================


def test_p0_1_neutral_snapshot_gives_zero_weights_everywhere(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 2, 100.0), **companion_flat_bars(2)},
        snapshots=[make_snapshot()],  # zero funding, empty options surface (stub sleeve, direction 0)
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    res = store.step_simulation_tick()
    assert res.bar_processed
    assert all(w == 0.0 for w in res.weights.values())
    assert all(t.weight == 0.0 for t in store.latest_targets.values())
    telemetry = store.sleeve_telemetry
    assert telemetry["funding_carry"]["is_actionable"] is False
    assert telemetry["derivatives_alpha"]["primary_direction"] == 0
    for coin in ASSETS:
        assert telemetry["funding_carry"]["evaluations"][coin]["is_actionable"] is False
        assert telemetry["derivatives_alpha"]["signals"][coin]["primary_direction"] == 0


def test_p0_1_ten_bps_funding_makes_carry_actionable_and_visible(tmp_path: Path) -> None:
    funding_1h = 0.000125  # 10 bps per 8h on BTC only
    snap = make_snapshot(funding_1h={"BTC": funding_1h})
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 2, 100.0, funding_1h=funding_1h), **companion_flat_bars(2)},
        snapshots=[snap],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    # Telemetry is live immediately after construction (no tick yet).
    tele = store.sleeve_telemetry["funding_carry"]
    assert tele["is_actionable"] is True
    assert tele["active_strategy"] == "SPOT_LONG_PERP_SHORT"
    assert tele["annualized_apr_pct"] == pytest.approx(funding_1h * 24 * 365 * 100.0, rel=1e-6)
    assert tele["evaluations"]["BTC"]["is_actionable"] is True
    assert tele["evaluations"]["ETH"]["is_actionable"] is False
    assert tele["cross_venue_spread_bps"] is None  # no second venue feed: flagged, not invented

    res = store.step_simulation_tick()
    assert res.bar_processed
    carry_key = position_key(SLEEVE_FUNDING_CARRY, "BTC")
    assert store.latest_targets[carry_key].weight > 0.0
    assert store.latest_targets[carry_key].hedged is True
    pos = store.position_manager.open_positions[carry_key]
    assert pos.hedged is True and pos.side == -1  # spot long / perp short
    assert store.sleeve_telemetry["funding_carry"]["status"] == "HARVESTING_ACTIVE"
    # Directional weights stay zero: the carry leg pair is delta neutral.
    assert res.weights == {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}


# =============================================================================
# P0.3: real HRP-CVaR weights
# =============================================================================


def test_p0_3_hrp_weights_follow_return_matrix_and_sum_to_one() -> None:
    def matrix(vol_btc: float, vol_eth: float, vol_sol: float) -> np.ndarray:
        rows = []
        for h in range(200):
            rows.append(
                [
                    vol_btc * math.sin(h / 5.0),
                    vol_eth * math.sin(h / 7.0) + 0.001 * math.cos(h / 3.0),
                    vol_sol * math.cos(h / 11.0),
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    sched = HRPWeightScheduler(("BTC", "ETH", "SOL"))
    w1 = sched.compute(0, matrix(0.004, 0.009, 0.002), force=True)  # SOL calmest, ETH wildest
    w2 = sched.compute(3_600_000_000_000, matrix(0.012, 0.002, 0.006), force=True)  # ETH calmest, BTC wildest
    assert sum(w1.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(w2.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(w > 0.0 for w in w1.values())
    assert w1 != w2
    assert any(abs(w1[a] - w2[a]) > 0.01 for a in w1)
    assert sched.last_result_is_fallback is False

    assert w1["SOL"] > w1["ETH"] and w2["ETH"] > w2["BTC"]  # lower CVaR earns the larger risk budget

    # Hourly cache: same hour returns the cached weights even when the matrix changes.
    w3 = sched.compute(3_600_000_000_000 + 10, matrix(0.004, 0.009, 0.002))
    assert w3 == w2


def test_p0_3_fewer_than_96_rows_falls_back_to_equal_weights_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    sched = HRPWeightScheduler(("BTC", "ETH", "SOL"), min_rows=96)
    small = np.full((40, 3), 0.001, dtype=np.float64)
    with caplog.at_level("WARNING", logger="phoenix.hrp"):
        w = sched.compute(0, small, force=True)
    assert w == pytest.approx({"BTC": 1 / 3, "ETH": 1 / 3, "SOL": 1 / 3})
    assert sched.last_result_is_fallback is True
    assert any("fallback to equal weights" in rec.message for rec in caplog.records)


def test_p0_3_rolling_hourly_matrix_appends_once_per_hour() -> None:
    m = RollingHourlyReturnMatrix(("BTC", "ETH", "SOL"), max_rows=10)
    hour = 3_600_000_000_000
    assert m.update(0, {"BTC": 100.0, "ETH": 10.0, "SOL": 5.0}) is False
    assert m.update(hour // 2, {"BTC": 101.0, "ETH": 10.0, "SOL": 5.0}) is False  # same hour, no row
    assert m.update(hour + 1, {"BTC": 102.0, "ETH": 9.0, "SOL": 5.5}) is True
    assert m.n_rows == 1
    row = m.matrix()[0]
    assert row[0] == pytest.approx(math.log(102.0 / 101.0))
    assert row[1] == pytest.approx(math.log(9.0 / 10.0))
    assert m.update(2 * hour + 1, {"BTC": 102.0}) is False  # incomplete universe ignored


def test_p0_3_store_uses_real_hrp_from_hourly_history(tmp_path: Path) -> None:
    hourly = {
        "BTC": hourly_closes_trending("BTC", 200, 100.0, slope_bps=2.0, wobble_bps=30.0),
        "ETH": hourly_closes_trending("ETH", 200, 10.0, slope_bps=-1.0, wobble_bps=80.0),
        "SOL": hourly_closes_trending("SOL", 200, 5.0, slope_bps=0.5, wobble_bps=150.0),
    }
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
        hourly_closes=hourly,
    )
    w = store.hrp_weights
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert store._hrp_scheduler.last_result_is_fallback is False
    assert w != pytest.approx({"BTC": 1 / 3, "ETH": 1 / 3, "SOL": 1 / 3})
    assert store.data_availability.hourly_return_rows == 199


# =============================================================================
# P0.4: real bar fields, no random generator in the tick path
# =============================================================================


def test_p0_4_no_numpy_random_in_tick_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Static check: the live loop module no longer references a random generator at all.
    src = (REPO_ROOT / "ui" / "api_server.py").read_text(encoding="utf-8")
    assert not re.search(r"np\.random|numpy\.random|default_rng|\b_rng\b|\.normal\(|\.uniform\(|\.exponential\(", src)
    pm_src = (REPO_ROOT / "core" / "execution" / "position_manager.py").read_text(encoding="utf-8")
    assert "random" not in pm_src

    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 2, 100.0), **companion_flat_bars(2)},
        snapshots=[make_snapshot(funding_1h={"BTC": 0.000125})],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("numpy.random was called inside the tick path")

    for name in ("default_rng", "normal", "uniform", "exponential", "rand", "randn", "random", "choice", "RandomState", "seed"):
        monkeypatch.setattr(np.random, name, _boom, raising=True)

    res = store.step_simulation_tick()
    assert res.bar_processed is True
    bar = store.history_bars[-1]
    assert bar.liquidation_volume_usd == 0.0
    assert store.data_availability.liquidations is False
    assert store.data_availability.taker_flow is False


def test_p0_4_candle_conversion_keeps_real_fields_and_flags_missing_ones() -> None:
    candle = {"t": BASE_TS_MS, "T": BASE_TS_MS + STEP_MS - 1, "o": "100.0", "h": "101.5", "l": "99.2", "c": "100.7", "v": "12.5", "n": 321}
    funding_hist = [(BASE_TS_MS - 7_200_000, 0.0002), (BASE_TS_MS - 3_600_000, 0.0003), (BASE_TS_MS + 3_600_000, 0.0009)]
    f1h = funding_rate_for_time(funding_hist, BASE_TS_MS)
    assert f1h == 0.0003  # most recent settlement at or before the candle open, never a later one
    bar = candle_to_dollar_bar(candle, instrument_id="BTCUSDT-PERP", horizon_label="15m", open_interest=4321.0, funding_rate_1h=f1h, mark_price=100.8, spot_price=100.6)
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 101.5, 99.2, 100.7)
    assert bar.volume == 12.5
    assert bar.tick_count == 321
    assert bar.dollar_volume == pytest.approx(12.5 * 100.7)
    assert bar.buy_dollar_volume == bar.sell_dollar_volume  # taker split unavailable: even split, flagged
    assert bar.liquidation_volume_usd == 0.0
    assert bar.funding_rate_8h == pytest.approx(0.0024)
    assert bar.ts_close_ns - bar.ts_open_ns == STEP_MS * 1_000_000
    avail = DataAvailability(candles=True, taker_flow=False, liquidations=False)
    assert avail.liquidations is False and avail.taker_flow is False


def test_p0_4_scripted_feed_cursor_returns_only_newer_closed_bars() -> None:
    bars = flat_bars("BTC", 0, 3, 100.0)
    feed = ScriptedBarFeed(bars_by_coin={"BTC": bars}, snapshots=[make_snapshot()])
    snap = feed.snapshot()
    first = feed.next_closed_bars(bars[0].ts_open_ns // 1_000_000, snap)
    assert first is not None and first["BTC"].ts_open_ns == bars[1].ts_open_ns
    second = feed.next_closed_bars(bars[1].ts_open_ns // 1_000_000, snap)
    assert second is not None and second["BTC"].ts_open_ns == bars[2].ts_open_ns
    assert feed.next_closed_bars(bars[2].ts_open_ns // 1_000_000, snap) is None


def test_p0_4_sigma_bar_is_ewma_of_15m_log_returns_span_96(tmp_path: Path) -> None:
    rets = [0.001 * math.sin(i / 4.0) for i in range(300)]
    series = compute_ewma_volatility(rets, span=96)
    alpha = 2.0 / 97.0
    var = rets[0] ** 2
    for i in range(1, len(rets)):
        var = (1 - alpha) * var + alpha * rets[i] ** 2
    assert series[-1] == pytest.approx(math.sqrt(var))
    est = EWMAVolatilityEstimator(span=96)
    est.seed(rets)
    assert est.value == pytest.approx(series[-1])
    assert est.is_warm

    history = sine_history("BTC", 200, 100.0)
    live = ohlc_path("BTC", [(100.0, 100.4, 99.6, 100.3), (100.3, 100.9, 100.1, 100.8)])
    store = build_store(tmp_path, live_bars={"BTC": live, **companion_flat_bars(2)}, snapshots=[make_snapshot()], history={"BTC": history})
    for _ in range(2):
        store.step_simulation_tick()
    expected = EWMAVolatilityEstimator.from_bars(history + live, span=96).value
    assert store.sigma_bar == pytest.approx(expected)
    assert store.sigma_bar != pytest.approx(0.0045)


# =============================================================================
# P0.5: position lifecycle
# =============================================================================


def _marks(bar: DollarBar, funding_1h: float = 0.0) -> dict[str, BarMarks]:
    return {"BTC": BarMarks(asset="BTC", bar=bar, mark_price=bar.close, spot_price=bar.close, funding_rate_1h=funding_1h)}


def test_p0_5_scripted_path_closes_exactly_three_trades_with_expected_pnl() -> None:
    fee = FeeSchedule.hyperliquid()
    pm = PositionManager(fee_schedule=fee, barrier_config=TripleBarrierConfig(fee_schedule=fee))
    target = SleeveTarget(sleeve_id=SLEEVE_DERIVATIVES, asset="BTC", weight=0.2, p_meta=0.8, reason="scripted")
    equity = 5000.0
    steps = barrier_path_steps()
    bars = ohlc_path("BTC", steps)

    # Bar 0: enter long at 100.00 (notional 1000).
    pos0 = pm.open_position(target, equity_usd=equity, price=bars[0].close, spot_price=bars[0].close, bar_index=0, ts_ns=bars[0].ts_close_ns, sigma_bar=1e-4)
    assert pos0 is not None
    assert pos0.upper_barrier == pytest.approx(100.8)  # +80 bps floor
    assert pos0.lower_barrier == pytest.approx(99.8)  # -20 bps floor
    assert pos0.vertical_bar_index == 16

    closed_all = []
    for i in range(1, len(bars)):
        closed = pm.process_bar(bar_index=i, ts_ns=bars[i].ts_close_ns, marks=_marks(bars[i]), targets={target.key: target})
        closed_all.extend(closed)
        # Win/loss counters move only on close.
        assert pm.wins + pm.losses == len(closed_all)
        if closed and i < len(bars) - 1:
            # Standing long signal: re-enter on the same bar once flat.
            pm.open_position(target, equity_usd=equity, price=bars[i].close, spot_price=bars[i].close, bar_index=i, ts_ns=bars[i].ts_close_ns, sigma_bar=1e-4)

    assert [c.exit_reason for c in closed_all] == [
        PositionExitReason.PROFIT_BARRIER,
        PositionExitReason.STOP_BARRIER,
        PositionExitReason.VERTICAL_EXPIRY,
    ]
    assert [c.exit_bar_index for c in closed_all] == [1, 2, 18]

    t1, t2, t3 = closed_all
    maker, taker = fee.maker_rate, fee.taker_rate
    # Trade 1: 10 units at 100 -> 100.80, maker in and out.
    assert t1.gross_pnl_usd == pytest.approx(8.0)
    assert t1.fee_usd == pytest.approx(1000.0 * maker + 1008.0 * maker)
    assert t1.net_pnl_usd == pytest.approx(8.0 - (1000.0 * maker + 1008.0 * maker))
    assert t1.is_win is True
    # Trade 2: entry 100.50, stop at -20 bps = 100.299, taker exit.
    size2 = 1000.0 / 100.5
    assert t2.entry_price == pytest.approx(100.5)
    assert t2.exit_price == pytest.approx(100.5 * (1.0 - 0.002))
    assert t2.gross_pnl_usd == pytest.approx(-2.0)
    assert t2.fee_usd == pytest.approx(1000.0 * maker + t2.exit_price * size2 * taker)
    assert t2.is_win is False
    # Trade 3: entry 100.20, flat until the 16th bar, exit at close 100.20 -> fees only.
    assert t3.entry_bar_index == 2 and t3.bars_held == 16
    assert t3.gross_pnl_usd == pytest.approx(0.0)
    assert t3.net_pnl_usd == pytest.approx(-(1000.0 * maker * 2.0))
    assert t3.is_win is False
    assert pm.wins == 1 and pm.losses == 2
    assert pm.realized_net_pnl_usd == pytest.approx(t1.net_pnl_usd + t2.net_pnl_usd + t3.net_pnl_usd)
    attribution = pm.sleeve_attribution()[SLEEVE_DERIVATIVES]
    assert attribution.closed_trades == 3 and attribution.wins == 1 and attribution.losses == 2


def test_p0_5_same_bar_double_touch_resolves_to_stop_and_regime_veto_exits() -> None:
    pm = PositionManager()
    target = SleeveTarget(sleeve_id=SLEEVE_DERIVATIVES, asset="BTC", weight=0.1, p_meta=0.7)
    pm.open_position(target, equity_usd=1000.0, price=100.0, spot_price=100.0, bar_index=0, ts_ns=0, sigma_bar=1e-4)
    wide = make_bar("BTC", 1, 100.0, 101.0, 99.0, 100.0)
    closed = pm.process_bar(bar_index=1, ts_ns=wide.ts_close_ns, marks=_marks(wide), targets={target.key: target})
    assert [c.exit_reason for c in closed] == [PositionExitReason.STOP_BARRIER]

    pm.open_position(target, equity_usd=1000.0, price=100.0, spot_price=100.0, bar_index=1, ts_ns=0, sigma_bar=1e-4)
    quiet = make_bar("BTC", 2, 100.0, 100.1, 99.9, 100.05)
    closed = pm.process_bar(bar_index=2, ts_ns=quiet.ts_close_ns, marks=_marks(quiet), targets={target.key: target}, regime_veto=True)
    assert [c.exit_reason for c in closed] == [PositionExitReason.REGIME_VETO]
    assert pm.open_positions == {}


def test_p0_5_flipped_target_exits_and_zero_target_holds_directional() -> None:
    pm = PositionManager()
    long_t = SleeveTarget(sleeve_id=SLEEVE_DERIVATIVES, asset="BTC", weight=0.1, p_meta=0.7)
    pm.open_position(long_t, equity_usd=1000.0, price=100.0, spot_price=100.0, bar_index=0, ts_ns=0, sigma_bar=1e-4)
    quiet = make_bar("BTC", 1, 100.0, 100.1, 99.9, 100.0)
    # Signal vanishing does not close a triple-barrier trade; the barriers own the exit.
    assert pm.process_bar(bar_index=1, ts_ns=1, marks=_marks(quiet), targets={}) == []
    short_t = long_t.model_copy(update={"weight": -0.1})
    closed = pm.process_bar(bar_index=2, ts_ns=2, marks=_marks(quiet), targets={short_t.key: short_t})
    assert [c.exit_reason for c in closed] == [PositionExitReason.TARGET_FLIPPED]


def test_p0_5_store_replay_records_three_closed_trades_and_counts_wins_only_on_close(tmp_path: Path) -> None:
    steps = barrier_path_steps()
    store = build_store(
        tmp_path,
        live_bars={"BTC": ohlc_path("BTC", steps), **companion_flat_bars(len(steps))},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
        derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.80, edge_bps=60.0),
    )
    key = position_key(SLEEVE_DERIVATIVES, "BTC")

    res0 = store.step_simulation_tick()
    assert res0.bar_processed and res0.forensic_record is not None
    assert res0.forensic_record["action"] == "ENTER_LONG"
    assert store.latest_targets[key].weight > 0.0
    pos = store.position_manager.open_positions[key]
    assert pos.entry_price == 100.0
    assert pos.upper_barrier == pytest.approx(100.8) and pos.lower_barrier == pytest.approx(99.8)
    assert store.ledger_summary["total_trades"] == 0  # open, not counted
    assert store.ledger_summary["win_count"] + store.ledger_summary["loss_count"] == 0

    for _ in range(len(steps) - 1):
        store.step_simulation_tick()

    trades = store.position_manager.closed_trades
    assert len(trades) == 3
    assert [t.exit_reason for t in trades] == [
        PositionExitReason.PROFIT_BARRIER,
        PositionExitReason.STOP_BARRIER,
        PositionExitReason.VERTICAL_EXPIRY,
    ]
    summary = store.ledger_summary
    assert summary["total_trades"] == 3
    assert summary["win_count"] == 1 and summary["loss_count"] == 2
    assert summary["trade_count"] == 3
    assert summary["total_realized_pnl"] == pytest.approx(round(sum(t.net_pnl_usd for t in trades), 2), abs=0.02)
    assert summary["total_fees_paid_usd"] == pytest.approx(round(sum(t.fee_usd for t in trades), 2), abs=0.02)
    n1 = trades[0].notional_usd
    assert trades[0].gross_pnl_usd == pytest.approx(n1 * 0.008)
    assert trades[1].gross_pnl_usd == pytest.approx(-trades[1].notional_usd * 0.002)
    assert trades[2].gross_pnl_usd == pytest.approx(0.0)
    # One re-entry stands open after the vertical exit; it is not in the counters.
    assert summary["open_positions"] == 1
    disk = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert disk["total_trades"] == 3 and len(disk["trade_history"]) == 3 and len(disk["open_positions"]) == 1
    assert disk["trade_history"][0]["exit_reason"] == "PROFIT_BARRIER"


# =============================================================================
# P0.6: signed, conditional funding and real fees
# =============================================================================


def test_p0_6_carry_perp_short_is_debited_when_funding_is_negative() -> None:
    pm = PositionManager()
    carry = SleeveTarget(sleeve_id=SLEEVE_FUNDING_CARRY, asset="BTC", weight=0.2, hedged=True, p_meta=0.8)
    pos = pm.open_position(carry, equity_usd=5000.0, price=100.0, spot_price=100.0, bar_index=0, ts_ns=0, sigma_bar=1e-4)
    assert pos is not None and pos.hedged and pos.side == -1 and pos.upper_barrier is None
    # Entry fees on both legs' notional.
    assert pos.entry_fee_usd == pytest.approx(1000.0 * 0.00015 * 2.0)

    bar = make_bar("BTC", 1, 100.0, 100.2, 99.8, 100.0)
    pm.process_bar(bar_index=1, ts_ns=1, marks=_marks(bar, funding_1h=-0.0004), targets={carry.key: carry})
    open_pos = pm.open_positions[carry.key]
    assert open_pos.funding_accrued_usd == pytest.approx(-0.0004 * 0.25 * 1000.0)  # debit: short pays negative funding

    pm.process_bar(bar_index=2, ts_ns=2, marks=_marks(bar, funding_1h=+0.0008), targets={carry.key: carry})
    assert pm.open_positions[carry.key].funding_accrued_usd == pytest.approx((-0.0004 + 0.0008) * 0.25 * 1000.0)

    closed = pm.process_bar(bar_index=3, ts_ns=3, marks=_marks(bar, funding_1h=0.0), targets={})
    assert [c.exit_reason for c in closed] == [PositionExitReason.TARGET_CLEARED]
    assert closed[0].funding_pnl_usd == pytest.approx(0.0001 * 1000.0)
    assert closed[0].gross_pnl_usd == pytest.approx(0.0)  # basis unchanged: delta-neutral
    assert closed[0].fee_usd == pytest.approx(1000.0 * 0.00015 * 4.0)  # two legs in, two legs out, maker


def test_p0_6_funding_accrues_only_on_open_perp_legs_and_is_signed() -> None:
    pm = PositionManager()
    bar = make_bar("BTC", 1, 100.0, 100.1, 99.9, 100.0)
    # Nothing open: no funding anywhere.
    assert pm.process_bar(bar_index=1, ts_ns=1, marks=_marks(bar, funding_1h=0.001), targets={}) == []
    assert pm.total_funding_pnl_usd == 0.0

    long_t = SleeveTarget(sleeve_id=SLEEVE_DERIVATIVES, asset="BTC", weight=0.1, p_meta=0.7)
    pm.open_position(long_t, equity_usd=10_000.0, price=100.0, spot_price=100.0, bar_index=1, ts_ns=1, sigma_bar=1e-4)
    pm.process_bar(bar_index=2, ts_ns=2, marks=_marks(bar, funding_1h=0.001), targets={long_t.key: long_t})
    assert pm.open_positions[long_t.key].funding_accrued_usd == pytest.approx(-0.001 * 0.25 * 1000.0)  # long pays


def test_p0_6_store_level_carry_funding_debit_and_no_fixed_turnover_fees(tmp_path: Path) -> None:
    snap = make_snapshot(funding_1h={"BTC": 0.000125})  # sleeve sees +10 bps/8h and opens the carry
    live_btc = flat_bars("BTC", 0, 3, 100.0, funding_1h=-0.0004)  # realised settlements on the bars are negative
    store = build_store(
        tmp_path,
        live_bars={"BTC": live_btc, **companion_flat_bars(3)},
        snapshots=[snap],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    key = position_key(SLEEVE_FUNDING_CARRY, "BTC")
    store.step_simulation_tick()  # opens the hedged carry at bar 0 close
    pos = store.position_manager.open_positions[key]
    assert pos.hedged and pos.side == -1
    assert store.ledger_summary["total_fees_paid_usd"] == 0.0  # fees are realised on close, not on a turnover guess
    store.step_simulation_tick()
    assert store.position_manager.open_positions[key].funding_accrued_usd == pytest.approx(-0.0004 * 0.25 * pos.size * 100.0)
    assert store.state.pnl_metrics.fee_savings_usd == 0.0
    assert store.ledger_summary["total_fee_savings_usd"] == 0.0


# =============================================================================
# P0.7: honest retraining
# =============================================================================


def test_p0_7_single_class_labels_skip_retrain_and_keep_previous_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    sentinel = object()
    store._ml_worker = sentinel  # type: ignore[assignment]

    captured: list[np.ndarray] = []
    import ui.api_server as api_mod

    original_label_bars = api_mod.FeeAdjustedTripleBarrierLabeler.label_bars

    def _spy(self: object, bars: object, primary_directions: object, sigma_bars: object = None) -> list[object]:
        labels = original_label_bars(self, bars, primary_directions, sigma_bars)  # type: ignore[arg-type]
        captured.append(np.array([lbl.meta_label for lbl in labels], dtype=np.int64))
        return list(labels)

    monkeypatch.setattr(api_mod.FeeAdjustedTripleBarrierLabeler, "label_bars", _spy)

    flat_train = flat_history("BTC", MIN_TRAIN_LABELLED_BARS + 40, 100.0)  # all directions 0 -> all labels 0
    result = store._fit_worker_on_bars(flat_train)
    assert result is None
    assert "single-class" in store._last_retrain_message
    assert store._ml_worker is sentinel
    assert len(captured) == 1
    assert np.unique(captured[0]).size == 1  # labels were not mutated into a fake two-class vector


def test_p0_7_requires_500_labelled_bars_after_purging_vertical_barrier(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    short_train = sine_history("BTC", MIN_TRAIN_LABELLED_BARS + 10, 100.0)  # 510 - 16 purge < 500
    assert store._fit_worker_on_bars(short_train) is None
    assert "500 required" in store._last_retrain_message


def test_p0_7_retrain_is_daily_not_every_20_ticks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    calls: list[int] = []

    def _fake_fit(bars: object) -> None:
        calls.append(1)
        store._last_retrain_message = "skipped in test"
        return None

    monkeypatch.setattr(store, "_fit_worker_on_bars", _fake_fit)
    t0 = store._last_retrain_ts_ns
    assert t0 is not None
    for k in range(1, 60):
        store._maybe_retrain(t0 + k * STEP_MS * 1_000_000)  # 59 bars = 14.75 hours
    assert calls == []
    store._maybe_retrain(t0 + RETRAIN_INTERVAL_NS)
    assert calls == [1]
    assert store.ml_worker is None  # previous (absent) worker retained on skip


def test_p0_7_two_class_real_history_trains_and_predicts_in_tick(tmp_path: Path) -> None:
    history = sine_history("BTC", MIN_TRAIN_LABELLED_BARS + 120, 100.0)
    live = ohlc_path("BTC", [(100.0, 100.3, 99.7, 100.2), (100.2, 100.5, 99.9, 100.1)])
    store = build_store(
        tmp_path,
        live_bars={"BTC": live, **companion_flat_bars(2)},
        snapshots=[make_snapshot()],
        history={"BTC": history},
        availability=DataAvailability(candles=True, funding_history=True, open_interest_history=True, feed_source="SCRIPTED"),
    )
    assert store.ml_worker is not None, store._last_retrain_message
    assert "fitted on" in store._last_retrain_message
    res = store.step_simulation_tick()
    assert res.bar_processed and res.forensic_record is not None
    meta = store.state.meta_predictions[-1]
    assert 0.0 <= meta.p_meta <= 1.0
    assert meta.svd_factors  # a real prediction, not the neutral placeholder
    assert store.latest_targets[position_key("meta_labeller", "BTC")].p_meta == pytest.approx(meta.p_meta)


# =============================================================================
# Phase 0 gate
# =============================================================================


def test_phase0_gate_zero_signal_replay_leaves_ledger_unchanged(tmp_path: Path) -> None:
    n = 40
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, n, 100.0), **companion_flat_bars(n)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
    )
    ledger_path = tmp_path / "ledger.json"
    before = ledger_core_fields(json.loads(ledger_path.read_text(encoding="utf-8")))
    for _ in range(n):
        res = store.step_simulation_tick()
        assert res.bar_processed
        assert res.forensic_record is not None and res.forensic_record["action"] == "HOLD_CASH"
    after = ledger_core_fields(json.loads(ledger_path.read_text(encoding="utf-8")))
    assert after == before
    assert after["total_trades"] == 0 and after["trade_history_len"] == 0
    assert store.tick_counter == 64 + n
    assert len(store.forensics_history) == n  # decisions are logged, trades are not invented


def test_phase0_gate_scripted_signals_produce_expected_trades_and_live_sleeve_panel(tmp_path: Path) -> None:
    steps = barrier_path_steps()
    funding_1h = 0.000125
    store = build_store(
        tmp_path,
        live_bars={"BTC": ohlc_path("BTC", steps, funding_1h=funding_1h), **companion_flat_bars(len(steps))},
        snapshots=[make_snapshot(funding_1h={"BTC": funding_1h})],
        history={"BTC": flat_history("BTC", 64, 100.0)},
        derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.80, edge_bps=60.0),
    )
    for _ in range(len(steps)):
        store.step_simulation_tick()
    closed = store.position_manager.closed_trades
    directional = [t for t in closed if t.sleeve_id == SLEEVE_DERIVATIVES]
    assert [t.exit_reason for t in directional] == [
        PositionExitReason.PROFIT_BARRIER,
        PositionExitReason.STOP_BARRIER,
        PositionExitReason.VERTICAL_EXPIRY,
    ]
    carry_key = position_key(SLEEVE_FUNDING_CARRY, "BTC")
    carry_pos = store.position_manager.open_positions[carry_key]
    assert carry_pos.hedged and carry_pos.funding_accrued_usd > 0.0  # short perp receives positive funding

    attribution = store.position_manager.sleeve_attribution(store._current_marks())
    assert attribution[SLEEVE_DERIVATIVES].closed_trades == 3
    assert attribution[SLEEVE_FUNDING_CARRY].open_positions == 1
    assert attribution[SLEEVE_FUNDING_CARRY].closed_trades == 0

    panel = store.sleeve_telemetry
    assert panel["funding_carry"]["status"] == "HARVESTING_ACTIVE"
    assert panel["funding_carry"]["annualized_apr_pct"] == pytest.approx(funding_1h * 24 * 365 * 100.0, rel=1e-6)
    assert panel["funding_carry"]["breakeven_hours"] is not None and panel["funding_carry"]["breakeven_hours"] > 0.0
    assert panel["derivatives_alpha"]["primary_direction"] == 1
    assert panel["derivatives_alpha"]["p_meta"] == pytest.approx(0.80)
    assert panel["derivatives_alpha"]["deribit_25d_skew"] == pytest.approx(0.02)
    assert panel["sleeve_attribution"][SLEEVE_DERIVATIVES]["closed_trades"] == 3
    assert panel["hrp_is_fallback"] is True  # no hourly history in this script: honest fallback
    assert panel["data_availability"]["liquidations"] is False
