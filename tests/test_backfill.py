"""P1.2 backfill tests: parsers on recorded fixtures, assembly, merge, coverage, and one live smoke."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import urllib.error

import polars as pl
import pytest

from data.backfill_storage import (
    BASE_COLUMNS,
    EXTRA_COLUMNS,
    MAX_ALLOWED_GAP_BARS,
    Candle,
    assemble_bars,
    backfill_asset,
    coverage_report,
    merge_into_parquet,
    parquet_path_for,
    parse_bybit_klines,
    parse_bybit_open_interest,
    parse_bybit_price_klines,
    parse_hyperliquid_funding,
    parse_okx_candles,
    parse_okx_open_interest,
    parse_okx_price_candles,
)

T0 = 1_760_000_000_000
STEP = 900_000

# Recorded venue payloads (field order as documented by each venue).
OKX_CANDLE_ROWS = [
    ["1760139900000", "114483.8", "114498.3", "112733.1", "112776.1", "312164.71", "3121.6471", "354179812.31431", "1"],
    ["1760139000000", "114300.0", "114600.0", "114200.0", "114483.8", "100000.0", "1000.0", "114400000.0", "1"],
    ["1760140800000", "112776.1", "112900.0", "112700.0", "112800.0", "50000.0", "500.0", "56400000.0", "0"],
]
BYBIT_KLINE_ROWS = [
    ["1760139900000", "114483.8", "114498.3", "112733.1", "112776.1", "3121.6471", "354179812.31"],
    ["1760139000000", "114300.0", "114600.0", "114200.0", "114483.8", "1000.0", "114400000.0"],
]
BYBIT_OI_ROWS = [
    {"openInterest": "41066.527", "timestamp": "1760139900000"},
    {"openInterest": "41000.000", "timestamp": "1760139000000"},
]
OKX_OI_ROWS = [["1793473200000", "3419929.12", "34199.2912", "2650947797.58"], ["1793469600000", "3419479.95", "34194.7995", "2652565825.77"]]
HL_FUNDING_ROWS = [
    {"coin": "BTC", "fundingRate": "0.0000125", "premium": "0.000163656", "time": 1793455200056},
    {"coin": "BTC", "fundingRate": "0.0000200", "premium": "0.0001", "time": 1793458800070},
]


def test_parsers_sort_oldest_first_and_keep_real_fields() -> None:
    okx = parse_okx_candles(OKX_CANDLE_ROWS)
    assert [c.t_ms for c in okx] == [1760139000000, 1760139900000]  # unconfirmed candle dropped
    assert okx[1].volume_base == pytest.approx(3121.6471) and okx[1].quote_volume == pytest.approx(354179812.31431)
    assert parse_okx_candles(OKX_CANDLE_ROWS, require_confirmed=False)[-1].t_ms == 1760140800000

    bybit = parse_bybit_klines(BYBIT_KLINE_ROWS)
    assert [c.t_ms for c in bybit] == [1760139000000, 1760139900000]
    assert bybit[1].open == 114483.8 and bybit[1].low == 112733.1 and bybit[1].volume_base == pytest.approx(3121.6471)

    assert parse_bybit_open_interest(BYBIT_OI_ROWS) == [(1760139000000, 41000.0), (1760139900000, 41066.527)]
    assert parse_okx_open_interest(OKX_OI_ROWS)[0] == (1793469600000, pytest.approx(34194.7995))
    assert parse_hyperliquid_funding(HL_FUNDING_ROWS) == [(1793455200056, 0.0000125), (1793458800070, 0.00002)]
    assert parse_bybit_price_klines([["1760139900000", "1", "2", "0.5", "1.5"]]) == [(1760139900000, 1.5)]
    assert parse_okx_price_candles([["1760139900000", "1", "2", "0.5", "1.7", "1"], ["1760140800000", "1", "2", "0.5", "1.9", "0"]]) == [(1760139900000, 1.7)]


def _candles(n: int, start_ms: int = T0) -> list[Candle]:
    out = []
    px = 100.0
    for i in range(n):
        c = px + 0.1 * (i % 5)
        out.append(Candle(t_ms=start_ms + i * STEP, open=px, high=max(px, c) + 0.05, low=min(px, c) - 0.05, close=c, volume_base=10.0 + i, quote_volume=(10.0 + i) * c))
        px = c
    return out


def test_assemble_bars_is_causal_and_flags_unavailable_fields() -> None:
    candles = _candles(8)
    funding = [(T0 - 3_600_000, 0.0001), (T0 + 2 * STEP, 0.0003)]  # settlement lands exactly at bar 2 open
    oi = [(T0 + i * STEP + STEP, 1000.0 + i) for i in range(5)]  # observed at bar closes for bars 0..4 only
    marks = [(T0 + i * STEP, 100.5 + i) for i in range(8)]
    df = assemble_bars("BTC", "15m", candles, funding=funding, open_interest=oi, mark_closes=marks, candle_source="okx", oi_source="bybit")
    assert df.height == 8
    assert set(BASE_COLUMNS) <= set(df.columns) and set(EXTRA_COLUMNS) <= set(df.columns)
    f1h = df.get_column("funding_rate_1h").to_list()
    assert f1h[:2] == [0.0001, 0.0001] and f1h[2:] == [0.0003] * 6  # never a later settlement
    assert df.get_column("funding_rate_8h").to_list()[3] == pytest.approx(0.0024)
    oi_col = df.get_column("open_interest").to_list()
    assert oi_col[:5] == [1000.0, 1001.0, 1002.0, 1003.0, 1004.0]
    assert oi_col[5] == 1004.0 and oi_col[6] == 1004.0  # within two bars: carried
    assert oi_col[7] is None  # staler than two bars: null, never invented
    assert df.get_column("mark_price").to_list()[3] == pytest.approx(103.5)
    assert df.get_column("spot_price").to_list()[3] == df.get_column("close").to_list()[3]  # no index feed: close
    assert df.get_column("liquidation_volume_usd").to_list() == [0.0] * 8
    assert df.get_column("taker_buy_volume").to_list() == [None] * 8
    assert df.get_column("buy_dollar_volume").to_list()[1] == pytest.approx(df.get_column("sell_dollar_volume").to_list()[1])
    assert df.get_column("bar_source")[0] == "BACKFILL-OKX" and df.get_column("oi_source")[0] == "BYBIT"
    flags = df.get_column("fields_unavailable")[0]
    assert "taker_flow" in flags and "liquidations" in flags and "spot_price" in flags and "mark_price" not in flags
    assert df.get_column("ts_close_ns").to_list()[0] - df.get_column("ts_open_ns").to_list()[0] == STEP * 1_000_000


def test_coverage_report_counts_rows_and_flags_gaps_longer_than_two_bars() -> None:
    candles = _candles(40)
    full = assemble_bars("BTC", "15m", candles)
    rep = coverage_report(full, "BTC", "15m")
    assert rep.rows == 40 and rep.expected_rows == 40 and rep.missing_rows == 0 and rep.max_gap_bars == 0
    assert rep.first_ts_ms == T0 and rep.last_ts_ms == T0 + 39 * STEP
    assert rep.passes_gap_rule

    holed = assemble_bars("BTC", "15m", [c for i, c in enumerate(candles) if i not in (10, 11, 12, 30)])
    rep2 = coverage_report(holed, "BTC", "15m")
    assert rep2.rows == 36 and rep2.missing_rows == 4
    assert rep2.max_gap_bars == 3 > MAX_ALLOWED_GAP_BARS
    assert rep2.gaps == ((T0 + 9 * STEP, 3), (T0 + 29 * STEP, 1))
    assert not rep2.passes_gap_rule


def test_merge_prefers_rows_with_open_interest_and_never_drops_live_rows(tmp_path: Path) -> None:
    path = tmp_path / "btc.parquet"
    live = assemble_bars("BTC", "15m", _candles(4), open_interest=[(T0 + i * STEP + STEP, 500.0) for i in range(4)], candle_source="live")
    assert merge_into_parquet(live, path) == 4
    backfill_without_oi = assemble_bars("BTC", "15m", _candles(6), candle_source="okx")
    assert merge_into_parquet(backfill_without_oi, path) == 6
    merged = pl.read_parquet(path).sort("ts_close_ns")
    assert merged.get_column("open_interest").to_list()[:4] == [500.0] * 4  # live rows kept
    assert merged.get_column("bar_source").to_list()[:4] == ["BACKFILL-LIVE"] * 4
    assert merged.get_column("open_interest").to_list()[4:] == [None, None]
    backfill_with_oi = assemble_bars("BTC", "15m", _candles(6), open_interest=[(T0 + i * STEP + STEP, 700.0) for i in range(6)], candle_source="bybit")
    assert merge_into_parquet(backfill_with_oi, path) == 6
    merged2 = pl.read_parquet(path).sort("ts_close_ns")
    assert merged2.get_column("open_interest").to_list()[4:] == [700.0, 700.0]  # nulls replaced by real OI
    assert merged2.get_column("open_interest").to_list()[0] == 700.0  # newer real OI row wins over older real row


def test_parquet_paths_follow_the_existing_store_names(tmp_path: Path) -> None:
    assert parquet_path_for("BTC", "15m", tmp_path) == tmp_path / "btcusdt_perp_bars.parquet"
    assert parquet_path_for("eth", "1h", tmp_path) == tmp_path / "ethusdt_perp_bars_1h.parquet"


def test_live_backfill_two_days_of_btc_1h_from_okx_and_hyperliquid(tmp_path: Path) -> None:
    """Live smoke: 48 hours of BTC 1h bars with Hyperliquid funding; skipped when the network is unavailable."""
    end = datetime.now(timezone.utc) - timedelta(hours=2)
    start = end - timedelta(hours=48)
    try:
        df, rep = backfill_asset("BTC", "1h", int(start.timestamp() * 1000), int(end.timestamp() * 1000), data_dir=tmp_path, venues=("okx",), pause_sec=0.05)
    except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - network dependent
        pytest.skip(f"network unavailable: {exc}")
    if rep.rows == 0:  # pragma: no cover - network dependent
        pytest.skip("OKX returned no candles from this network")
    assert rep.rows >= 44 and rep.passes_gap_rule, rep.as_dict()
    assert rep.candle_source == "BACKFILL-OKX"
    assert rep.funding_rows > 0  # Hyperliquid funding attached
    assert (tmp_path / "btcusdt_perp_bars_1h.parquet").exists()
    assert df.get_column("mark_price").to_list() != df.get_column("close").to_list()  # real mark prices
