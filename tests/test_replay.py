"""P1.7 replay harness tests: scenarios, snapshot alignment, Parquet selection, and end-to-end runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import polars as pl
import pytest

from core.contracts import PositionExitReason
from research.replay import (
    NAMED_SCENARIOS,
    ReplayScenario,
    ScenarioData,
    hourly_closes_from_bars,
    load_scenario_bars,
    run_replay,
    snapshots_for_replay,
    summarize,
)
from tests.phase0_helpers import (
    BASE_TS_MS,
    STEP_MS,
    ForcedSignalDerivativesSleeve,
    barrier_path_steps,
    companion_flat_bars,
    flat_bars,
    flat_history,
    ohlc_path,
    sine_history,
)


def test_named_scenarios_cover_the_four_plan_windows() -> None:
    assert set(NAMED_SCENARIOS) == {"calm_2025_09", "cascade_2025_10_10", "tariff_2025_02_03", "carry_unwind_2024_08_05"}
    for sc in NAMED_SCENARIOS.values():
        assert sc.start_utc < sc.end_utc
        assert sc.start_utc.tzinfo is not None
    assert NAMED_SCENARIOS["cascade_2025_10_10"].start_utc <= datetime(2025, 10, 10, tzinfo=timezone.utc) <= NAMED_SCENARIOS["cascade_2025_10_10"].end_utc


def _scenario_for(live_bars: int, history_bars: int) -> ReplayScenario:
    start = datetime.fromtimestamp(BASE_TS_MS / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp((BASE_TS_MS + live_bars * STEP_MS) / 1000, tz=timezone.utc)
    return ReplayScenario(name="unit", start_utc=start, end_utc=end, warmup_bars=history_bars)


def test_snapshots_align_to_primary_bars_and_carry_bar_marks() -> None:
    steps = barrier_path_steps()
    live = {"BTC": ohlc_path("BTC", steps, funding_1h=0.0002), **companion_flat_bars(len(steps))}
    live["SOL"] = live["SOL"][:-3]  # SOL misses the last three bars: last seen value carries
    data = ScenarioData(history_by_coin={"BTC": flat_history("BTC", 20, 100.0)}, live_by_coin=live, hourly_closes_by_coin={})
    snaps = snapshots_for_replay(data)
    assert len(snaps) == len(steps) + 1  # one for construction, one per BTC bar
    assert snaps[0].get_asset("BTC") is not None and snaps[0].timestamp_ns == data.history_by_coin["BTC"][-1].ts_close_ns
    for i, bar in enumerate(live["BTC"]):
        s = snaps[i + 1]
        btc = s.get_asset("BTC")
        assert btc is not None and btc.mid_price == bar.close and btc.funding_rate_1h == pytest.approx(0.0002)
        assert s.timestamp_ns == bar.ts_close_ns
    sol_last = snaps[-1].get_asset("SOL")
    assert sol_last is not None and sol_last.mid_price == live["SOL"][-1].close


def test_hourly_closes_take_the_last_close_of_each_hour() -> None:
    bars = flat_bars("BTC", 0, 8, 100.0)
    bars = [b.model_copy(update={"close": 100.0 + i}) for i, b in enumerate(bars)]
    closes = hourly_closes_from_bars(bars)
    assert len(closes) == 2
    assert closes[0][1] == 103.0 and closes[1][1] == 107.0


def test_load_scenario_bars_selects_window_and_warmup_from_parquet(tmp_path: Path) -> None:
    history = sine_history("BTC", 300, 100.0)
    live = ohlc_path("BTC", barrier_path_steps())
    rows = [dict(b.model_dump(), bar_source="BACKFILL-OKX") for b in history + live]
    rows[5]["open_interest"] = None  # a backfilled row without OI: forward-filled for replay, excluded from training
    path = tmp_path / "btcusdt_perp_bars.parquet"
    pl.DataFrame(rows).write_parquet(path)
    sc = _scenario_for(live_bars=len(live), history_bars=100).model_copy(update={"coins": ("BTC",)})
    data = load_scenario_bars(sc, tmp_path)
    assert len(data.live_by_coin["BTC"]) == len(live)
    assert len(data.history_by_coin["BTC"]) == 100
    assert data.history_by_coin["BTC"][-1].ts_close_ns <= sc.start_ns < data.live_by_coin["BTC"][0].ts_close_ns
    assert len(data.training_bars) == 299  # the null-OI row is not a training row
    assert data.hourly_closes_by_coin["BTC"]


def test_run_replay_reproduces_the_scripted_trades_and_writes_a_report(tmp_path: Path) -> None:
    steps = barrier_path_steps()
    live = {"BTC": ohlc_path("BTC", steps), **companion_flat_bars(len(steps))}
    data = ScenarioData(history_by_coin={"BTC": flat_history("BTC", 64, 100.0)}, live_by_coin=live, hourly_closes_by_coin={})
    sc = _scenario_for(live_bars=len(steps), history_bars=64)
    report = run_replay(sc, data, report_dir=tmp_path / "replays", derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.8, edge_bps=60.0))
    assert report.bars_processed == report.bars_available == len(steps)
    assert report.n_trades == 3 and report.wins == 1 and report.losses == 2
    assert [t["exit_reason"] for t in report.trades] == [
        PositionExitReason.PROFIT_BARRIER.value,
        PositionExitReason.STOP_BARRIER.value,
        PositionExitReason.VERTICAL_EXPIRY.value,
    ]
    assert report.open_positions_at_end == 1
    assert len(report.equity_curve) == len(steps)
    assert report.turnover_usd > 0.0 and report.fees_usd > 0.0
    assert report.final_equity_usd == pytest.approx(report.starting_capital_usd + report.net_pnl_usd, abs=0.02)
    assert report.max_drawdown_usd >= 0.0
    assert report.markouts_available is False
    assert report.sleeve_attribution["derivatives_alpha"]["closed_trades"] == 3
    assert report.elapsed_seconds < 120.0
    assert report.report_path is not None and Path(report.report_path).exists()
    on_disk = json.loads(Path(report.report_path).read_text(encoding="utf-8"))
    assert on_disk["scenario_name"] == "unit" and on_disk["n_trades"] == 3
    assert "trades 3" in summarize(report)


def test_run_replay_with_no_signal_changes_nothing(tmp_path: Path) -> None:
    n = 30
    live = {"BTC": flat_bars("BTC", 0, n, 100.0), **companion_flat_bars(n)}
    data = ScenarioData(history_by_coin={"BTC": flat_history("BTC", 64, 100.0)}, live_by_coin=live, hourly_closes_by_coin={})
    report = run_replay(_scenario_for(n, 64), data, report_dir=None)
    assert report.bars_processed == n and report.n_trades == 0
    assert report.net_pnl_usd == 0.0 and report.fees_usd == 0.0 and report.turnover_usd == 0.0
    assert report.final_equity_usd == report.starting_capital_usd
    assert report.report_path is None
    assert all(pt.open_positions == 0 for pt in report.equity_curve)
