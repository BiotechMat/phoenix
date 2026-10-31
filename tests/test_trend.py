"""P2.2 tests: trend forecasts, sizing, causality, target exit policy, store wiring (off by default), evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from core.contracts import PositionExitReason
from core.execution.position_manager import BarMarks, PositionManager, SleeveTarget
from research.trend_eval import evaluate_trend
from research.trial_registry import TrialRegistry
from sleeves.sleeve_2_trend import (
    TREND_SLEEVE_ID,
    TrendSleeve,
    TrendSleeveConfig,
    backtest_trend,
    lookback_forecasts,
    rolling_average_pairwise_correlation,
)
from tests.phase0_helpers import build_store, companion_flat_bars, flat_bars, flat_history, hourly_closes_trending, make_bar, make_snapshot


def _series(n: int, drift: float, wobble: float, period: float, base: float) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return base * np.exp(drift * i + wobble * np.sin(i / period))


def test_lookback_forecast_matches_definition() -> None:
    closes = _series(200, 0.001, 0.004, 3.0, 100.0)
    lc = np.log(closes)
    f = lookback_forecasts(lc, 24)
    r = np.diff(lc[-25:])
    expected = math.tanh((lc[-1] - lc[-25]) / (float(np.std(r, ddof=1)) * math.sqrt(24)))
    assert f[-1] == pytest.approx(expected)
    assert np.all(np.isnan(f[:24])) and np.all(np.abs(f[24:]) <= 1.0)


def test_backtest_is_causal_and_signs_follow_trends() -> None:
    n = 1600
    up = _series(n, 0.0006, 0.004, 5.0, 100.0)
    down = _series(n, -0.0006, 0.004, 7.0, 50.0)
    flat = _series(n, 0.0, 0.004, 11.0, 20.0)
    bt = backtest_trend(np.arange(n), {"UP": up, "DOWN": down, "FLAT": flat})
    cfg = TrendSleeveConfig()
    assert np.all(bt.weights[: cfg.min_history_hours] == 0.0)
    assert bt.weights[-1, 0] > 0.0 and bt.weights[-1, 1] < 0.0
    assert np.all(np.abs(bt.weights) <= cfg.max_weight_per_asset + 1e-12)
    # Changing the future must not change any earlier weight.
    up2 = up.copy()
    up2[1200:] *= 0.5
    bt2 = backtest_trend(np.arange(n), {"UP": up2, "DOWN": down, "FLAT": flat})
    assert np.array_equal(bt.weights[:1200], bt2.weights[:1200])
    assert np.isnan(bt.simple_returns_next[-1]).all()


def test_inverse_vol_sizing_and_correlation_adjustment() -> None:
    n = 1600
    calm = _series(n, 0.0006, 0.002, 5.0, 100.0)
    wild = _series(n, 0.0006, 0.010, 5.0, 100.0) * _series(n, 0.0, 0.003, 13.0, 1.0)
    other = _series(n, 0.0, 0.004, 17.0, 10.0)
    cfg = TrendSleeveConfig(max_weight_per_asset=2.5, rebalance_buffer=0.0)
    bt = backtest_trend(np.arange(n), {"CALM": calm, "WILD": wild, "OTHER": other}, cfg)
    assert abs(bt.weights[-1, 0]) > abs(bt.weights[-1, 1])  # lower vol, larger weight

    lc = np.log(np.column_stack([calm, calm * 1.5, calm * 2.0]))  # identical returns: correlation 1
    rho = rolling_average_pairwise_correlation(lc, 720)
    assert rho[-1] == pytest.approx(1.0)
    assert 1.0 / math.sqrt(1.0 + 2.0 * rho[-1]) == pytest.approx(1.0 / math.sqrt(3.0))


def test_rebalance_buffer_suppresses_small_weight_changes() -> None:
    n = 1600
    up = _series(n, 0.0006, 0.004, 5.0, 100.0)
    other = _series(n, 0.0, 0.004, 9.0, 30.0)
    loose = backtest_trend(np.arange(n), {"A": up, "B": other}, TrendSleeveConfig(rebalance_buffer=0.0))
    tight = backtest_trend(np.arange(n), {"A": up, "B": other}, TrendSleeveConfig(rebalance_buffer=0.05))
    changes_loose = int(np.sum(np.abs(np.diff(loose.weights, axis=0)) > 0))
    changes_tight = int(np.sum(np.abs(np.diff(tight.weights, axis=0)) > 0))
    assert changes_tight < changes_loose
    moves = np.abs(np.diff(tight.weights, axis=0))
    assert np.all((moves == 0.0) | (moves > 0.05) | (tight.weights[1:] == 0.0))


def test_target_exit_policy_ignores_barriers_and_exits_on_clear_or_flip() -> None:
    pm = PositionManager()
    tgt = SleeveTarget(sleeve_id=TREND_SLEEVE_ID, asset="BTC", weight=0.2, exit_policy="target", p_meta=0.7)
    pos = pm.open_position(tgt, equity_usd=1000.0, price=100.0, spot_price=100.0, bar_index=0, ts_ns=0, sigma_bar=1e-4)
    assert pos is not None and pos.exit_policy == "target" and pos.upper_barrier is None and pos.vertical_bar_index is None
    big_move = make_bar("BTC", 1, 100.0, 110.0, 90.0, 100.0)
    marks = {"BTC": BarMarks(asset="BTC", bar=big_move, mark_price=100.0, spot_price=100.0)}
    for k in range(1, 40):  # far past 16 bars, through a 10% range: still open while the target stands
        assert pm.process_bar(bar_index=k, ts_ns=k, marks=marks, targets={tgt.key: tgt}) == []
    flipped = tgt.model_copy(update={"weight": -0.2})
    closed = pm.process_bar(bar_index=40, ts_ns=40, marks=marks, targets={tgt.key: flipped})
    assert [c.exit_reason for c in closed] == [PositionExitReason.TARGET_FLIPPED]
    pm.open_position(tgt, equity_usd=1000.0, price=100.0, spot_price=100.0, bar_index=41, ts_ns=41, sigma_bar=1e-4)
    closed = pm.process_bar(bar_index=42, ts_ns=42, marks=marks, targets={})
    assert [c.exit_reason for c in closed] == [PositionExitReason.TARGET_CLEARED]


def _trend_hourly() -> dict[str, list[tuple[int, float]]]:
    return {
        "BTC": hourly_closes_trending("BTC", 800, 100.0, slope_bps=8.0, wobble_bps=20.0),
        "ETH": hourly_closes_trending("ETH", 800, 10.0, slope_bps=-8.0, wobble_bps=20.0),
        "SOL": hourly_closes_trending("SOL", 800, 5.0, slope_bps=0.5, wobble_bps=60.0),
    }


def test_trend_sleeve_is_off_by_default(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0), **companion_flat_bars(1)},
        snapshots=[make_snapshot()],
        history={"BTC": flat_history("BTC", 64, 100.0)},
        hourly_closes=_trend_hourly(),
    )
    store.step_simulation_tick()
    assert not any(k.startswith(f"{TREND_SLEEVE_ID}:") for k in store.latest_targets)
    assert store.sleeve_telemetry["trend"]["enabled"] is False


def test_enabled_trend_sleeve_opens_target_positions_from_hourly_history(tmp_path: Path) -> None:
    from core.data_ingestion import ScriptedBarFeed
    from ui.api_server import ControlCenterStateStore

    hourly = _trend_hourly()
    last_px = {c: v[-1][1] for c, v in hourly.items()}
    feed = ScriptedBarFeed(
        bars_by_coin={c: flat_bars(c, 0, 2, last_px[c]) for c in ("BTC", "ETH", "SOL")},
        snapshots=[make_snapshot(prices=last_px)],
        history_by_coin={"BTC": flat_history("BTC", 64, last_px["BTC"])},
        hourly_closes_by_coin=hourly,
    )
    store = ControlCenterStateStore(
        ledger_path=tmp_path / "l.json",
        parquet_path=tmp_path / "b.parquet",
        feed=feed,
        fetch_derivatives_inputs=False,
        trend_sleeve=TrendSleeve(TrendSleeveConfig(enabled=True)),
    )
    store.step_simulation_tick()
    btc = store.latest_targets[f"{TREND_SLEEVE_ID}:BTC"]
    eth = store.latest_targets[f"{TREND_SLEEVE_ID}:ETH"]
    assert btc.weight > 0.0 and eth.weight < 0.0 and btc.exit_policy == "target"
    pos = store.position_manager.open_positions[f"{TREND_SLEEVE_ID}:BTC"]
    assert pos.exit_policy == "target" and pos.upper_barrier is None
    tele = store.sleeve_telemetry["trend"]
    assert tele["enabled"] is True and tele["open_positions"] >= 2 and tele["signals"]["BTC"]["forecast"] > 0.0
    store.step_simulation_tick()  # a quiet bar: trend positions stay open
    assert f"{TREND_SLEEVE_ID}:BTC" in store.position_manager.open_positions


def test_evaluate_trend_reports_validation_crisis_coverage_and_logs_trials(tmp_path: Path) -> None:
    n = 2500
    ts = np.arange(n, dtype=np.int64) * 3_600_000 + 1_600_000_000_000  # 2020-09: outside every crisis window
    closes = {
        "BTC": _series(n, 0.0004, 0.006, 6.0, 100.0),
        "ETH": _series(n, -0.0003, 0.006, 8.0, 10.0),
        "SOL": _series(n, 0.0, 0.01, 10.0, 5.0),
    }
    funding = {c: np.full(n, 0.0000125) for c in closes}
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    report = evaluate_trend(ts, closes, funding, registry=reg, cpcv_groups=6, cpcv_test_groups=2, report_dir=tmp_path)
    assert report.n_hours == n - 1 and report.assets == ("BTC", "ETH", "SOL")
    assert len(report.per_asset) == 3 and reg.count("trend") == 3
    assert [a.total_trials_used for a in report.per_asset] == [1, 2, 3]
    assert report.fees_total > 0.0
    assert report.crisis_scenarios_covered == 0 and all(v is None for v in report.crisis_pnl.values())
    assert report.passes_crisis_rule is False  # cannot pass without covering the crises
    assert report.derivatives_correlation is None
    assert report.report_path is not None and Path(report.report_path).exists()
