"""P1.4 / P0.8 tests: endpoint parsers on recorded fixtures, TTL cache, stablecoin marks, macro inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts import DataAvailability
from core.market_feeds import (
    DvolSnapshot,
    HyperliquidAuxSnapshot,
    MacroInputs,
    PredictedFunding,
    ScriptedHyperliquidAuxFeed,
    ScriptedMacroInputsFeed,
    StablecoinQuote,
    TTLCache,
    compute_dvol_snapshot,
    compute_stablecoin_marks,
    instruments_for_tenor,
    parse_dvol_series,
    parse_instruments,
    parse_oi_cap_assets,
    parse_option_ticker,
    parse_perp_dex_limits,
    parse_predicted_fundings,
)
from tests.phase0_helpers import build_store, companion_flat_bars, flat_bars, flat_history, make_snapshot

# Recorded payloads (31 Oct 2026).
PREDICTED_FUNDINGS = [
    ["BTC", [["BinPerp", {"fundingRate": "0.00005", "nextFundingTime": 1793491200000, "fundingIntervalHours": 4}],
             ["HlPerp", {"fundingRate": "0.0000125", "nextFundingTime": 1793476800000, "fundingIntervalHours": 1}],
             ["BybitPerp", {"fundingRate": "0.0001", "nextFundingTime": 1793491200000, "fundingIntervalHours": 8}]]],
    ["ETH", [["HlPerp", {"fundingRate": "-0.00002", "nextFundingTime": 1793476800000, "fundingIntervalHours": 1}]]],
    ["BROKEN", "not-a-list"],
]
OI_CAPS = ["CANTO", "FTM", "JELLY", "LOOM", "RLB", "SOPH", "VINE", "ZEREBRO"]
DEX_LIMITS = {"totalOiCap": "1000000000.0", "oiSzCapPerPerp": "25000000.0", "maxTransferNtl": "500000.0",
              "coinToOiCap": [["xyz:AAPL", "200000000.0"], ["xyz:ACN", "25000000.0"]]}
DVOL_PAYLOAD = {"jsonrpc": "2.0", "result": {"data": [[1793469600000, 39.99, 40.03, 39.98, 40.03], [1793473200000, 40.03, 40.21, 40.03, 40.09],
                                                      [1793476800000, 40.09, 40.1, 40.09, 60.0]], "continuation": None}}
INSTRUMENTS = {"result": [
    {"instrument_name": "BTC-27NOV26-77000-C", "expiration_timestamp": 1795852800000, "strike": 77000.0, "option_type": "call", "settlement_period": "month"},
    {"instrument_name": "BTC-27NOV26-77000-P", "expiration_timestamp": 1795852800000, "strike": 77000.0, "option_type": "put", "settlement_period": "month"},
    {"instrument_name": "BTC-1NOV26-70000-C", "expiration_timestamp": 1793520000000, "strike": 70000.0, "option_type": "call", "settlement_period": "day"},
    {"instrument_name": "BTC-29JAN27-80000-C", "expiration_timestamp": 1801296000000, "strike": 80000.0, "option_type": "call", "settlement_period": "month"},
    {"instrument_name": "BTC-PERPETUAL", "kind": "future"},
]}
TICKER = {"result": {"timestamp": 1793478228452, "greeks": {"delta": 0.55927, "gamma": 5e-05, "vega": 82.6589, "theta": -59.99492, "rho": 28.94995},
                     "index_price": 77467.62, "open_interest": 310.1, "mark_price": 0.0463, "mark_iv": 38.44, "bid_iv": 38.17, "ask_iv": 38.64,
                     "underlying_price": 77780.19, "underlying_index": "BTC-27NOV26", "instrument_name": "BTC-27NOV26-77000-C"}}


def test_parse_hyperliquid_predicted_fundings_and_normalisation() -> None:
    pf = parse_predicted_fundings(PREDICTED_FUNDINGS)
    assert set(pf) == {"BTC", "ETH"}  # malformed row dropped
    btc = pf["BTC"]
    assert btc["HlPerp"].rate_per_hour == pytest.approx(0.0000125)
    assert btc["BinPerp"].rate_per_hour == pytest.approx(0.00005 / 4.0)
    assert btc["BybitPerp"].rate_per_day == pytest.approx(0.0001 / 8.0 * 24.0)
    assert btc["BinPerp"].next_funding_ms == 1793491200000
    assert pf["ETH"]["HlPerp"].rate_per_day == pytest.approx(-0.00048)
    assert parse_predicted_fundings({"unexpected": 1}) == {}


def test_parse_oi_caps_and_dex_limits() -> None:
    caps = parse_oi_cap_assets(OI_CAPS)
    assert "JELLY" in caps and "BTC" not in caps and len(caps) == 8
    assert parse_oi_cap_assets(None) == frozenset()
    lim = parse_perp_dex_limits("xyz", DEX_LIMITS)
    assert lim is not None and lim.total_oi_cap_usd == 1_000_000_000.0 and lim.oi_size_cap_per_perp == 25_000_000.0
    assert lim.asset_streaming_oi_caps_usd["xyz:AAPL"] == 200_000_000.0
    assert parse_perp_dex_limits("", None) is None  # the main dex reports null: no limits, not zero limits


def test_parse_dvol_series_and_zscore() -> None:
    series = parse_dvol_series(DVOL_PAYLOAD)
    assert [c for _, c in series] == [40.03, 40.09, 60.0]
    snap = compute_dvol_snapshot(series, "btc", window_hours=720)
    assert snap is not None and snap.currency == "BTC" and snap.index == 60.0 and snap.n_obs == 3
    expected_z = (60.0 - (40.03 + 40.09) / 2.0) / (abs(40.03 - 40.09) / 2.0)
    assert snap.zscore == pytest.approx(expected_z)
    assert compute_dvol_snapshot([], "BTC") is None
    assert compute_dvol_snapshot([(1, 50.0)], "BTC") is not None and compute_dvol_snapshot([(1, 50.0)], "BTC").zscore == 0.0  # type: ignore[union-attr]


def test_parse_instruments_tenor_selection_and_ticker_greeks() -> None:
    inst = parse_instruments(INSTRUMENTS)
    assert [i.instrument_name for i in inst] == ["BTC-1NOV26-70000-C", "BTC-27NOV26-77000-C", "BTC-27NOV26-77000-P", "BTC-29JAN27-80000-C"]
    now_ms = 1793478228452
    thirty = instruments_for_tenor(inst, now_ms, target_days=30.0, tolerance_days=7.0)
    assert {i.instrument_name for i in thirty} == {"BTC-27NOV26-77000-C", "BTC-27NOV26-77000-P"}
    assert instruments_for_tenor(inst, now_ms, target_days=180.0, tolerance_days=7.0) == []
    t = parse_option_ticker(TICKER)
    assert t is not None
    assert t.instrument_name == "BTC-27NOV26-77000-C" and t.mark_iv == 38.44 and t.delta == pytest.approx(0.55927)
    assert t.gamma == 5e-05 and t.underlying_price == 77780.19 and t.open_interest == 310.1
    assert parse_option_ticker({"error": {"code": -32602}}) is None


def test_ttl_cache_refreshes_after_ttl_and_serves_stale_on_failure() -> None:
    cache = TTLCache()
    calls: list[int] = []

    def loader() -> int:
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("venue down")
        return len(calls)

    assert cache.get("k", ttl_ns=100, loader=loader, now_ns=0) == 1
    assert cache.get("k", ttl_ns=100, loader=loader, now_ns=50) == 1  # within TTL: no reload
    assert cache.get("k", ttl_ns=100, loader=loader, now_ns=150) == 2
    assert cache.get("k", ttl_ns=100, loader=loader, now_ns=300) == 2  # loader failed: stale value served
    assert cache.last_error("k") == "venue down"
    assert cache.get("missing", ttl_ns=100, loader=lambda: (_ for _ in ()).throw(RuntimeError("x")), now_ns=0) is None


def test_stablecoin_marks_need_two_venues_and_convert_usdt_crosses() -> None:
    quotes = [
        StablecoinQuote(symbol="USDT", venue="coinbase", price=0.9989),
        StablecoinQuote(symbol="USDT", venue="kraken", price=0.9991),
        StablecoinQuote(symbol="USDC", venue="kraken", price=0.9998),
        StablecoinQuote(symbol="USDC", venue="okx", price=1.0009, quote_currency="USDT"),  # USDC/USDT cross
        StablecoinQuote(symbol="USDE", venue="kraken", price=0.9700),  # only one venue: not eligible
    ]
    marks = compute_stablecoin_marks(quotes, ts_ns=7)
    assert marks.median_usd["USDT"] == pytest.approx(0.9990)
    assert marks.venues_per_symbol == {"USDT": 2, "USDC": 2, "USDE": 1}
    assert marks.median_usd["USDC"] == pytest.approx((0.9998 + 1.0009 * 0.9990) / 2.0)
    assert marks.depeg_bps["USDE"] == pytest.approx(300.0)
    assert marks.max_depeg_bps == pytest.approx(max(marks.depeg_bps["USDT"], marks.depeg_bps["USDC"]))  # USDE excluded
    assert marks.symbols_with_two_venues == ["USDC", "USDT"]

    # Without a USD quote for USDT the cross cannot be converted and is dropped, not assumed at par.
    marks2 = compute_stablecoin_marks([StablecoinQuote(symbol="USDC", venue="okx", price=1.0009, quote_currency="USDT")])
    assert marks2.median_usd == {} and marks2.max_depeg_bps is None


def test_availability_record_lists_every_feed() -> None:
    flags = DataAvailability(candles=True, dvol=True).live_feeds()
    assert set(flags) >= {"candles", "funding_history", "open_interest_history", "predicted_fundings", "oi_caps", "options_surface", "hlp_vault", "dvol", "stablecoin_marks", "taker_flow", "liquidations", "trades", "lead_mids"}
    assert flags["candles"] is True and flags["dvol"] is True and flags["lead_mids"] is False


# ---------------------------------------------------------------------------------------------
# P0.8: live inputs for the risk committee
# ---------------------------------------------------------------------------------------------


def _store_with_macro(tmp_path: Path, inputs: MacroInputs | None, aux: HyperliquidAuxSnapshot | None = None):  # type: ignore[no-untyped-def]
    from ui.api_server import ControlCenterStateStore
    from core.data_ingestion import ScriptedBarFeed

    feed = ScriptedBarFeed(
        bars_by_coin={"BTC": flat_bars("BTC", 0, 2, 100.0), **companion_flat_bars(2)},
        snapshots=[make_snapshot()],
        history_by_coin={"BTC": flat_history("BTC", 64, 100.0)},
    )
    return ControlCenterStateStore(
        ledger_path=tmp_path / "ledger.json",
        parquet_path=tmp_path / "bars.parquet",
        feed=feed,
        fetch_derivatives_inputs=False,
        macro_feed=ScriptedMacroInputsFeed(inputs) if inputs is not None else None,
        aux_feed=ScriptedHyperliquidAuxFeed(aux) if aux is not None else None,
    )


def test_p0_8_sentinel_depeg_and_dvol_come_from_fetched_marks(tmp_path: Path) -> None:
    quotes = [
        StablecoinQuote(symbol="USDT", venue="coinbase", price=0.9960),
        StablecoinQuote(symbol="USDT", venue="kraken", price=0.9964),
        StablecoinQuote(symbol="USDC", venue="kraken", price=0.9999),
        StablecoinQuote(symbol="USDC", venue="bitstamp", price=1.0001),
    ]
    marks = compute_stablecoin_marks(quotes)
    dvol = DvolSnapshot(currency="BTC", index=71.5, zscore=2.4, n_obs=720, window_hours=720, ts_ms=1)
    inputs = MacroInputs(stablecoin_depeg_bps=marks.max_depeg_bps, dvol_index=dvol.index, dvol_zscore=dvol.zscore, stablecoin_marks=marks, dvol=dvol)
    store = _store_with_macro(tmp_path, inputs)

    sentinel = store.risk_committee_output.macro_sentinel
    assert sentinel.stablecoin_depeg_bps == pytest.approx(marks.max_depeg_bps)
    assert sentinel.stablecoin_depeg_bps == pytest.approx(38.0)  # USDT median 0.9962 -> 38 bps, not the old 1.5 constant
    assert sentinel.dvol_index == pytest.approx(71.5) and sentinel.dvol_zscore == pytest.approx(2.4)
    assert store.data_availability.stablecoin_marks is True and store.data_availability.dvol is True

    res = store.step_simulation_tick()
    assert res.bar_processed
    after = store.risk_committee_output.macro_sentinel
    assert after.stablecoin_depeg_bps == pytest.approx(38.0) and after.dvol_index == pytest.approx(71.5)
    assert store.sleeve_telemetry["macro_inputs"]["stablecoin_depeg_bps"] == pytest.approx(38.0)


def test_p0_8_missing_macro_feeds_contribute_no_stress_and_are_flagged(tmp_path: Path) -> None:
    store = _store_with_macro(tmp_path, MacroInputs())  # everything None
    sentinel = store.risk_committee_output.macro_sentinel
    assert sentinel.stablecoin_depeg_bps == 0.0
    assert sentinel.dvol_index == pytest.approx(52.0)  # the sentinel's own neutral default, flagged below
    assert store.data_availability.stablecoin_marks is False and store.data_availability.dvol is False
    assert store.data_availability.live_feeds()["dvol"] is False


def test_p1_4_hyperliquid_aux_feed_reaches_telemetry_and_availability(tmp_path: Path) -> None:
    aux = HyperliquidAuxSnapshot(
        predicted_fundings={"BTC": {"HlPerp": PredictedFunding(venue="HlPerp", funding_rate=0.0000125, interval_hours=1.0, next_funding_ms=1),
                                    "BybitPerp": PredictedFunding(venue="BybitPerp", funding_rate=0.0001, interval_hours=8.0, next_funding_ms=1)}},
        oi_cap_assets=frozenset({"JELLY"}),
        predicted_fundings_available=True,
        oi_caps_available=True,
    )
    store = _store_with_macro(tmp_path, MacroInputs(), aux)
    tele = store.sleeve_telemetry["hyperliquid_aux"]
    assert tele["oi_cap_assets"] == ["JELLY"]
    assert tele["predicted_funding_per_day"]["BTC"]["HlPerp"] == pytest.approx(0.0000125 * 24.0)
    assert tele["predicted_funding_per_day"]["BTC"]["BybitPerp"] == pytest.approx(0.0001 / 8.0 * 24.0)
    assert store.data_availability.predicted_fundings is True and store.data_availability.oi_caps is True
