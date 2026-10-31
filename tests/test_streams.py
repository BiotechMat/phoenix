"""P1.1 / P1.3 / P1.5 tests: stream parsers, buffers, boundary imbalance and store integration (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from core.contracts import PositioningRegime
from core.data_ingestion import ScriptedBarFeed
from core.streams import (
    BybitTopOfBook,
    Coverage,
    LeadMidBuffer,
    LiquidationEvent,
    StreamHub,
    Trade,
    TradeTape,
    boundary_imbalance_from_tape,
    compute_boundary_imbalance,
    parse_binance_book_ticker,
    parse_bybit_all_liquidation,
    parse_hyperliquid_trades,
    parse_okx_liquidation_orders,
    quarter_hour_boundaries,
    replay_recorded_messages,
)
from tests.phase0_helpers import BASE_TS_MS, STEP_MS, companion_flat_bars, flat_bars, flat_history, make_bar, make_snapshot
from ui.api_server import ControlCenterStateStore

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "stream_recording_btc.json"

OKX_LIQUIDATIONS = {"code": "0", "msg": "", "data": [{"details": [
    {"bkLoss": "0", "bkPx": "77723.9", "ccy": "", "posSide": "short", "side": "buy", "sz": "15.51", "time": 1793469242359, "ts": "1793469242359"},
    {"bkLoss": "0", "bkPx": "77720", "ccy": "", "posSide": "long", "side": "sell", "sz": "7.51", "time": 1793469240093, "ts": "1793469240093"}],
    "instFamily": "BTC-USDT", "instId": "BTC-USDT-SWAP", "instType": "SWAP", "totalLoss": "0", "uly": "BTC-USDT"}]}
BYBIT_LIQUIDATION = {"topic": "allLiquidation.BTCUSDT", "type": "snapshot", "ts": 1739502303204,
                     "data": [{"T": 1739502302929, "s": "BTCUSDT", "S": "Buy", "v": "0.5", "p": "96000"},
                              {"T": 1739502302950, "s": "BTCUSDT", "S": "Sell", "v": "0.25", "p": "96100"}]}


def _recording() -> dict[str, list[dict[str, object]]]:
    data: dict[str, list[dict[str, object]]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


# ------------------------------------------------------------------------------------------------
# Parsers on recorded payloads
# ------------------------------------------------------------------------------------------------


def test_parse_recorded_hyperliquid_trades() -> None:
    msgs = [row["raw"] for row in _recording()["hyperliquid_trades"]]
    parsed = [p for m in msgs if isinstance(m, dict) for p in parse_hyperliquid_trades(m)]
    assert parsed, "the recording contains at least one trade"
    coin, trade = parsed[0]
    assert coin == "BTC" and trade.price > 1000.0 and trade.size > 0.0 and trade.side in (1, -1)
    assert parse_hyperliquid_trades({"channel": "subscriptionResponse", "data": {}}) == []
    synthetic = {"channel": "trades", "data": [{"coin": "ETH", "side": "A", "px": "2000", "sz": "1.5", "time": 5}]}
    assert parse_hyperliquid_trades(synthetic) == [("ETH", Trade(ts_ms=5, price=2000.0, size=1.5, side=-1))]


def test_parse_recorded_binance_and_bybit_top_of_book() -> None:
    rec = _recording()
    binance = [parse_binance_book_ticker(r["raw"]) for r in rec["binance_bookticker"] if isinstance(r["raw"], dict)]
    binance_ok = [b for b in binance if b is not None]
    assert len(binance_ok) > 100
    assert all(c == "BTC" for c, _, _ in binance_ok)
    lo = min(m for _, _, m in binance_ok)
    hi = max(m for _, _, m in binance_ok)
    book = BybitTopOfBook()
    bybit_ok = [x for x in (book.update(r["raw"]) for r in rec["bybit_orderbook1"] if isinstance(r["raw"], dict)) if x is not None]
    assert bybit_ok and all(c == "BTC" for c, _, _ in bybit_ok)
    for _, _, mid in bybit_ok:
        assert lo * 0.999 <= mid <= hi * 1.001  # the two venues agree within 10 bps
    wrapped = {"stream": "btcusdt@bookTicker", "data": {"e": "bookTicker", "s": "BTCUSDT", "b": "100", "a": "102", "T": 9}}
    assert parse_binance_book_ticker(wrapped) == ("BTC", 9, 101.0)


def test_bybit_top_of_book_merges_deltas() -> None:
    book = BybitTopOfBook()
    assert book.update({"topic": "orderbook.1.BTCUSDT", "type": "snapshot", "ts": 1, "data": {"s": "BTCUSDT", "b": [["100", "1"]], "a": [["102", "1"]]}}) == ("BTC", 1, 101.0)
    assert book.update({"topic": "orderbook.1.BTCUSDT", "type": "delta", "ts": 2, "data": {"s": "BTCUSDT", "b": [], "a": [["101", "2"]]}}) == ("BTC", 2, 100.5)
    assert book.update({"topic": "orderbook.1.BTCUSDT", "type": "delta", "ts": 3, "data": {"s": "BTCUSDT", "b": [["100", "0"]], "a": []}}) is None


def test_parse_liquidations_bybit_and_okx() -> None:
    by = parse_bybit_all_liquidation(BYBIT_LIQUIDATION)
    assert [(e.position_side, e.size_base, e.price) for e in by] == [("long", 0.5, 96000.0), ("short", 0.25, 96100.0)]
    assert by[0].notional_usd == 48000.0 and by[0].venue == "bybit"
    okx = parse_okx_liquidation_orders(OKX_LIQUIDATIONS, {"BTC-USDT-SWAP": 0.01})
    assert [(e.position_side, e.size_base) for e in okx] == [("long", pytest.approx(0.0751)), ("short", pytest.approx(0.1551))]
    assert okx[1].notional_usd == pytest.approx(0.1551 * 77723.9)
    assert parse_okx_liquidation_orders(OKX_LIQUIDATIONS, {}) == []  # unknown contract size: dropped, not guessed


# ------------------------------------------------------------------------------------------------
# P1.5 lead mids on the recorded 10-second window
# ------------------------------------------------------------------------------------------------


def test_p1_5_recorded_window_lead_move_matches_independent_computation() -> None:
    rec = _recording()
    rows = [parse_binance_book_ticker(r["raw"]) for r in rec["binance_bookticker"] if isinstance(r["raw"], dict)]
    series = [(ts, mid) for x in rows if x is not None for _, ts, mid in [x]]
    span_ms = series[-1][0] - series[0][0]
    assert span_ms >= 8_000  # a real ~10-second window

    buf = LeadMidBuffer(retention_ms=20_000)
    for ts, mid in series:
        buf.add("binance", "BTC", ts, mid)
    for window in (700, 2_000, 5_000):
        last_ts, last_mid = series[-1]
        base = [m for t, m in series if t <= last_ts - window][-1]
        expected = (last_mid - base) / base * 10_000.0
        assert buf.lead_mid_change_bps("BTC", window, venue="binance") == pytest.approx(expected)

    # Production retention keeps only ~2 s, enough for the 700 ms trigger, not for a 5 s window.
    hub = StreamHub(("BTC",))
    replay_recorded_messages(hub, rec)
    assert hub.lead_mids.lead_mid_change_bps("BTC", 700, venue="binance") is not None
    assert hub.lead_mids.lead_mid_change_bps("BTC", 5_000, venue="binance") is None
    assert "binance" in hub.lead_mids.venues("BTC")


def test_lead_mid_median_across_venues_and_out_of_order_updates() -> None:
    buf = LeadMidBuffer(retention_ms=5_000)
    for ts, a, b in [(0, 100.0, 200.0), (500, 100.5, 200.0), (1000, 101.0, 202.0)]:
        buf.add("binance", "BTC", ts, a)
        buf.add("bybit", "BTC", ts, b)
    buf.add("binance", "BTC", 900, 50.0)  # out of order: ignored
    assert buf.lead_mid_change_bps("BTC", 1000, venue="binance") == pytest.approx(100.0)
    assert buf.lead_mid_change_bps("BTC", 1000) == pytest.approx(100.0)  # median of 100 and 100
    assert buf.lead_mid_change_bps("BTC", 5_000) is None


# ------------------------------------------------------------------------------------------------
# Coverage and P1.1 boundary imbalance
# ------------------------------------------------------------------------------------------------


def test_coverage_requires_continuous_stream() -> None:
    cov = Coverage(max_silence_ms=100)
    for t in range(0, 1001, 50):
        cov.alive(t)
    assert cov.covers(0, 1000)
    cov.break_()
    cov.alive(1100)
    assert not cov.covers(900, 1200)  # break between 1000 and 1100
    cov.alive(1150)
    cov.alive(1400)  # silence of 250 ms > 100: new interval
    assert cov.covers(1100, 1150) and not cov.covers(1100, 1400)
    assert cov.covered_ms_within(0, 2000) == 1000 + 50


def _tape_with_trades(boundary: int) -> TradeTape:
    tape = TradeTape(("BTC",), max_silence_ms=10_000_000)
    # Trailing 2 hours: 1 BTC every minute, alternating sides (volume 120 over 7200 s).
    trailing = [Trade(ts_ms=boundary - 7_200_000 + i * 60_000, price=100.0, size=1.0, side=1 if i % 2 else -1) for i in range(120)]
    tape.add("BTC", trailing)
    after = [
        Trade(ts_ms=boundary + 2_000, price=100.0, size=3.0, side=1),
        Trade(ts_ms=boundary + 9_999, price=100.0, size=1.0, side=-1),
        Trade(ts_ms=boundary + 15_000, price=100.0, size=2.0, side=1),
        Trade(ts_ms=boundary + 45_000, price=100.0, size=4.0, side=-1),
        Trade(ts_ms=boundary + 61_000, price=100.0, size=9.0, side=1),  # outside every window
    ]
    tape.add("BTC", after, received_ms=boundary + 61_000)
    return tape


def test_p1_1_boundary_imbalance_on_constructed_tape() -> None:
    boundary = (BASE_TS_MS // 900_000 + 10) * 900_000
    assert quarter_hour_boundaries(boundary - 1, boundary + 900_000) == [boundary, boundary + 900_000]
    tape = _tape_with_trades(boundary)
    rec = boundary_imbalance_from_tape(tape, "BTC", boundary)
    assert rec is not None
    assert rec.signed_volume == {10: 2.0, 30: 4.0, 60: 0.0}
    assert rec.total_volume == {10: 4.0, 30: 6.0, 60: 10.0}
    covered_s = (boundary - (boundary - 7_200_000)) / 1000.0  # trailing coverage starts at the first trade
    per_second = 120.0 / covered_s
    assert rec.normalized[10] == pytest.approx(2.0 / (per_second * 10))
    assert rec.normalized[30] == pytest.approx(4.0 / (per_second * 30))
    assert rec.normalized[60] == pytest.approx(0.0)
    assert rec.trailing_hours_used == pytest.approx(2.0, rel=1e-3)
    assert rec.n_trades == 4

    direct = compute_boundary_imbalance("BTC", boundary, tape.trades_between("BTC", boundary, boundary + 60_000), 120.0, int(covered_s * 1000))
    assert direct.normalized == rec.normalized


def test_p1_1_no_feature_when_stream_did_not_cover_the_window() -> None:
    boundary = (BASE_TS_MS // 900_000 + 10) * 900_000
    tape = TradeTape(("BTC",), max_silence_ms=10_000_000)
    tape.add("BTC", [Trade(ts_ms=boundary - 3_600_000 + i * 60_000, price=100.0, size=1.0, side=1) for i in range(60)])
    tape.add("BTC", [Trade(ts_ms=boundary + 5_000, price=100.0, size=1.0, side=1)], received_ms=boundary + 10_000)
    tape.break_coverage("BTC")  # disconnect 10 s after the boundary
    tape.add("BTC", [Trade(ts_ms=boundary + 40_000, price=100.0, size=1.0, side=-1)], received_ms=boundary + 70_000)
    assert boundary_imbalance_from_tape(tape, "BTC", boundary) is None  # gap inside the 60 s window
    assert tape.covers("BTC", boundary + 40_000, boundary + 70_000)


def test_p1_1_tracker_records_each_boundary_once_and_flushes_parquet(tmp_path: Path) -> None:
    boundary = (BASE_TS_MS // 900_000 + 10) * 900_000
    hub = StreamHub(("BTC",))
    hub.tape = _tape_with_trades(boundary)
    from core.streams import BoundaryImbalanceTracker

    hub.boundaries = BoundaryImbalanceTracker(hub.tape)
    first = hub.boundaries.update(boundary + 61_000)
    assert [r.boundary_ms for r in first] == [boundary - 900_000, boundary]  # the earlier boundary is covered too
    assert hub.boundaries.update(boundary + 120_000) == []  # each boundary is computed once
    latest = hub.boundaries.latest("BTC")
    assert latest is not None and latest.boundary_ms == boundary
    rows = hub.tape.flush_parquet(tmp_path)
    assert rows == 125
    files = list((tmp_path / "btc").glob("trades_*.parquet"))
    assert len(files) == 1 and pl.read_parquet(files[0]).height == 125
    assert hub.tape.flush_parquet(tmp_path) == 0  # nothing new


# ------------------------------------------------------------------------------------------------
# Store integration (P1.3 acceptance)
# ------------------------------------------------------------------------------------------------


def _hub_covering_bar(open_ms: int, close_ms: int, *, liquidations: list[LiquidationEvent], trades: list[Trade]) -> StreamHub:
    hub = StreamHub(("BTC", "ETH", "SOL"))
    for t in range(open_ms - 60_000, close_ms + 60_001, 30_000):
        hub.liquidations.heartbeat(t)
        hub.tape.heartbeat("BTC", t)
    hub.liquidations.add(liquidations)
    hub.tape.add("BTC", trades, received_ms=close_ms + 1)
    return hub


def test_p1_3_liquidation_flag_flips_and_bar_carries_real_liquidation_and_taker_volume(tmp_path: Path) -> None:
    live = flat_bars("BTC", 0, 1, 100.0)
    open_ms, close_ms = live[0].ts_open_ns // 1_000_000, live[0].ts_close_ns // 1_000_000
    liqs = [
        LiquidationEvent(ts_ms=open_ms + 10_000, coin="BTC", position_side="long", size_base=10.0, price=100.0, venue="bybit"),
        LiquidationEvent(ts_ms=open_ms + 20_000, coin="BTC", position_side="short", size_base=2.0, price=100.0, venue="bybit"),
        LiquidationEvent(ts_ms=close_ms + 5_000, coin="BTC", position_side="long", size_base=99.0, price=100.0, venue="bybit"),  # next bar
    ]
    trades = [Trade(ts_ms=open_ms + 1_000, price=100.0, size=3.0, side=1), Trade(ts_ms=open_ms + 2_000, price=100.0, size=1.0, side=-1)]
    hub = _hub_covering_bar(open_ms, close_ms, liquidations=liqs, trades=trades)
    feed = ScriptedBarFeed(bars_by_coin={"BTC": live, **companion_flat_bars(1)}, snapshots=[make_snapshot()], history_by_coin={"BTC": flat_history("BTC", 64, 100.0)})
    store = ControlCenterStateStore(ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False, stream_hub=hub)
    assert store.data_availability.liquidations is False  # nothing processed yet

    res = store.step_simulation_tick()
    assert res.bar_processed
    bar = store.history_bars[-1]
    assert bar.liquidation_volume_usd == pytest.approx(1200.0)
    assert bar.buy_dollar_volume == pytest.approx(300.0) and bar.sell_dollar_volume == pytest.approx(100.0)
    assert store.data_availability.liquidations is True and store.data_availability.taker_flow is True
    flow = store.stream_telemetry["assets"]["BTC"]["last_bar_flow"]
    assert flow["long_liquidation_usd"] == pytest.approx(1000.0) and flow["short_liquidation_usd"] == pytest.approx(200.0)
    # The liquidation stream covers every subscribed coin, so ETH's zero is an observed zero;
    # the trade tape only covered BTC, so ETH's taker flow stays unavailable.
    eth_flow = store.stream_telemetry["assets"]["ETH"]["last_bar_flow"]
    assert eth_flow["liquidation_usd"] == 0.0 and eth_flow["taker_buy_usd"] is None


def test_p1_3_cascade_exhaustion_regime_fires_on_streamed_liquidations(tmp_path: Path) -> None:
    history = flat_history("BTC", 120, 100.0)
    cascade = make_bar("BTC", 0, 100.0, 100.0, 96.0, 97.0, open_interest=900.0)  # OI flush of 10%, sharp selloff
    open_ms, close_ms = cascade.ts_open_ns // 1_000_000, cascade.ts_close_ns // 1_000_000
    liqs = [LiquidationEvent(ts_ms=open_ms + 60_000 + i * 1000, coin="BTC", position_side="long", size_base=50.0, price=97.0, venue="bybit") for i in range(40)]
    hub = _hub_covering_bar(open_ms, close_ms, liquidations=liqs, trades=[])
    feed = ScriptedBarFeed(bars_by_coin={"BTC": [cascade], **companion_flat_bars(1)}, snapshots=[make_snapshot()], history_by_coin={"BTC": history})
    store = ControlCenterStateStore(ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False, stream_hub=hub)
    store.step_simulation_tick()
    assert store.history_bars[-1].liquidation_volume_usd == pytest.approx(40 * 50.0 * 97.0)
    sig = store.state.positioning_signals[-1]
    assert sig.liquidation_exhaustion_flag == 1
    assert sig.regime == PositioningRegime.CASCADE_EXHAUSTION_REVERSAL
    assert sig.primary_direction == 1  # mean reversion after a long-liquidation selloff

    # The same bar without stream coverage stays at zero liquidations and does not classify as a cascade.
    feed2 = ScriptedBarFeed(bars_by_coin={"BTC": [cascade], **companion_flat_bars(1)}, snapshots=[make_snapshot()], history_by_coin={"BTC": history})
    store2 = ControlCenterStateStore(ledger_path=tmp_path / "l2.json", parquet_path=tmp_path / "b2.parquet", feed=feed2, fetch_derivatives_inputs=False, stream_hub=StreamHub(("BTC", "ETH", "SOL")))
    store2.step_simulation_tick()
    assert store2.history_bars[-1].liquidation_volume_usd == 0.0
    assert store2.state.positioning_signals[-1].regime != PositioningRegime.CASCADE_EXHAUSTION_REVERSAL
    assert store2.data_availability.liquidations is False

    _ = STEP_MS


def test_p1_1_tracker_retries_a_boundary_until_quiet_coin_coverage_catches_up() -> None:
    boundary = (BASE_TS_MS // 900_000 + 10) * 900_000
    tape = TradeTape(("SOL",), max_silence_ms=10_000_000)
    tape.add("SOL", [Trade(ts_ms=boundary - 600_000 + i * 60_000, price=5.0, size=1.0, side=1) for i in range(10)])
    tape.add("SOL", [Trade(ts_ms=boundary + 20_000, price=5.0, size=2.0, side=-1)])  # coverage ends at boundary + 20 s
    from core.streams import BoundaryImbalanceTracker

    tracker = BoundaryImbalanceTracker(tape)
    assert tracker.update(boundary + 65_000) == []  # not covered yet: retried later, not abandoned
    tape.heartbeat("SOL", boundary + 90_000)  # a pong / next trade extends coverage past the window
    recs = tracker.update(boundary + 95_000)
    assert [r.boundary_ms for r in recs] == [boundary]
    assert recs[0].signed_volume[30] == -2.0

    stale = TradeTape(("SOL",), max_silence_ms=10_000_000)
    stale.add("SOL", [Trade(ts_ms=boundary - 60_000, price=5.0, size=1.0, side=1)])
    stale_tracker = BoundaryImbalanceTracker(stale)
    assert stale_tracker.update(boundary + 60_000 + 300_000) == []  # retry budget spent: abandoned
    stale.heartbeat("SOL", boundary + 400_000)
    assert stale_tracker.update(boundary + 410_000) == []  # not computed retroactively from a gapped window
