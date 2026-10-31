"""P4.2 tests: markout ladder resolution, grouping, toxicity feedback, agreement with the simulator, store wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.execution.markouts import MarkoutTracker, MidHistory, minutes_to_hourly_funding, mids_from_mapping, size_bucket
from core.streams import StreamHub
from research.quote_sim import QuoteSimConfig, simulate_quotes
from tests.phase0_helpers import BASE_TS_MS, ForcedSignalDerivativesSleeve, companion_flat_bars, flat_bars, flat_history, make_snapshot
from tests.test_lead_pull import _lead_then_hl_drop


def test_buckets_and_funding_clock() -> None:
    assert size_bucket(500.0) == "<1,000" and size_bucket(5_000.0) == "<10,000" and size_bucket(1e6) == ">=100,000"
    assert minutes_to_hourly_funding(3_600_000 * 5) == 0.0
    assert minutes_to_hourly_funding(3_600_000 * 5 + 15 * 60_000) == pytest.approx(45.0)


def test_mid_history_staleness_and_retention() -> None:
    h = MidHistory(retention_ms=10_000, max_staleness_ms=2_000)
    for t in range(0, 20_001, 1_000):
        h.add("BTC", t, 100.0 + t / 1_000)
    assert h.mid_at("BTC", 15_500) == 115.0
    assert h.mid_at("BTC", 5_000) is None  # retained only 10 s back from the latest point
    assert h.mid_at("BTC", 25_000) is None  # beyond the last observation: unknown, not extrapolated
    sparse = MidHistory(max_staleness_ms=2_000)
    sparse.add("ETH", 0, 10.0)
    sparse.add("ETH", 10_000, 11.0)
    assert sparse.mid_at("ETH", 5_000) is None  # the only earlier mid is 5 s stale


def test_tracker_resolves_horizons_groups_and_tightens_toxicity() -> None:
    mids = mids_from_mapping({"BTC": [(t, 100.0 - 0.001 * t / 1_000) for t in range(0, 400_001, 500)]})
    tr = MarkoutTracker()
    tr.record_fill(fill_id="a", coin="BTC", side="BUY", price=100.0, ts_ms=0, notional_usd=500.0, sleeve_id="trend", lead_move_bps=-2.0)
    tr.record_fill(fill_id="b", coin="BTC", side="SELL", price=100.0, ts_ms=0, notional_usd=50_000.0, sleeve_id="carry", lead_move_bps=+0.1)
    assert tr.resolve(mids.mid_at, now_ms=4_000) == 2  # only the 1 s horizon is due
    tr.resolve(mids.mid_at, now_ms=400_000)
    buy, sell = tr.fills
    assert buy.markouts_bps[30_000] == pytest.approx(-0.03 / 100.0 * 1e4 * 1.0, rel=1e-6)  # price fell 0.03 over 30 s
    assert sell.markouts_bps[30_000] == pytest.approx(+0.03 / 100.0 * 1e4, rel=1e-6)
    assert buy.lead_state == "lead_against" and sell.lead_state == "lead_flat"
    by_sleeve = tr.summary("sleeve_id")
    assert set(by_sleeve) == {"trend", "carry"} and by_sleeve["trend"][300_000]["n"] == 1.0
    # The slow EWMA is seeded by the first fill (-3 bps) and moves 5% toward the second (+3 bps).
    assert tr.ewma_30s_bps == pytest.approx(-3.0 * 0.95 + 3.0 * 0.05)
    assert tr.toxicity_threshold(0.60) == pytest.approx(0.60 * (1.0 + (-2.7) / 20.0))

    bad = MarkoutTracker(ewma_alpha=1.0)
    bad.record_fill(fill_id="c", coin="BTC", side="BUY", price=100.0, ts_ms=0, notional_usd=1.0, sleeve_id="x")
    bad.resolve(mids_from_mapping({"BTC": [(30_000, 99.9)]}).mid_at, now_ms=31_000)
    assert bad.ewma_30s_bps == pytest.approx(-10.0)
    assert bad.toxicity_threshold(0.60) == pytest.approx(0.30)


def test_unavailable_mid_is_given_up_after_a_minute() -> None:
    tr = MarkoutTracker(horizons_ms=(5_000,))
    tr.record_fill(fill_id="d", coin="SOL", side="BUY", price=5.0, ts_ms=0, notional_usd=10.0, sleeve_id="x")
    empty = MidHistory()
    assert tr.resolve(empty.mid_at, now_ms=10_000) == 0
    assert tr.resolve(empty.mid_at, now_ms=70_000) == 1
    assert tr.fills[0].markouts_bps[5_000] is None


def test_tracker_agrees_with_simulator_markouts() -> None:
    events = _lead_then_hl_drop()
    sim = simulate_quotes(events, "BTC", QuoteSimConfig(lead_pull=False))
    mids = mids_from_mapping({"BTC": [(e["recv_ms"], (float(e["raw"]["data"]["bbo"][0]["px"]) + float(e["raw"]["data"]["bbo"][1]["px"])) / 2.0)
                                      for e in events if e["raw"].get("channel") == "bbo"]})
    tr = MarkoutTracker(horizons_ms=(1_000, 5_000, 10_000, 30_000))
    for f in sim.fills:
        tr.record_fill(fill_id=str(f.ts_ms), coin="BTC", side=f.side, price=f.price, ts_ms=f.ts_ms, notional_usd=1.0, sleeve_id="sim")
    tr.resolve(lambda c, t: mids.mid_at(c, t) if mids.mid_at(c, t) is not None else None, now_ms=10**9)
    for f, r in zip(sim.fills, tr.fills):
        for h in (5_000, 10_000):
            assert r.markouts_bps[h] == pytest.approx(f.markouts_bps[h])


def test_store_records_paper_fills_and_resolves_them_from_hub_mids(tmp_path: Path) -> None:
    from core.data_ingestion import ScriptedBarFeed
    from ui.api_server import ControlCenterStateStore

    hub = StreamHub(("BTC", "ETH", "SOL"))
    for t in range(0, 400_001, 1_000):
        hub.hl_mids.add("BTC", BASE_TS_MS + t, 100.0 + 0.0005 * t / 1_000)
    feed = ScriptedBarFeed(bars_by_coin={"BTC": flat_bars("BTC", 0, 1, 100.0), **companion_flat_bars(1)}, snapshots=[make_snapshot()], history_by_coin={"BTC": flat_history("BTC", 64, 100.0)})
    store = ControlCenterStateStore(
        ledger_path=tmp_path / "l.json", parquet_path=tmp_path / "b.parquet", feed=feed, fetch_derivatives_inputs=False,
        derivatives_sleeve=ForcedSignalDerivativesSleeve(asset="BTC", direction=1, p_meta=0.8, edge_bps=60.0), stream_hub=hub,
    )
    store.step_simulation_tick()
    assert len(store.markout_tracker.fills) == 1 and store.markout_tracker.fills[0].fill_id.startswith("open:derivatives_alpha_BTC")
    store.resolve_markouts(now_ms=BASE_TS_MS + 10**7)
    tele = store.markout_telemetry
    assert tele["fills"] == 1 and tele["mid_source"] == "hyperliquid_bbo_stream"
    assert tele["all"]["all"][30_000]["median"] == pytest.approx((100.015 - 100.0) / 100.0 * 1e4, rel=1e-6)
