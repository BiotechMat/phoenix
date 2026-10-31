"""P4.1 tests: lead-venue pull rule, defensive-maker action and counter, quote simulator on constructed events."""

from __future__ import annotations

from typing import Any

from core.contracts import ExecutionAction
from core.execution.defensive_maker import DefensiveMakerExecutionAlgorithm, should_pull_for_lead_move
from research.quote_sim import QuoteSimConfig, simulate_quotes


def test_pull_rule_is_side_aware_and_scaled_by_spread() -> None:
    assert should_pull_for_lead_move("BUY", -1.0, 1.0, k=0.5) is True  # lead fell 1 bp > 0.5 x 1 bp
    assert should_pull_for_lead_move("BUY", -0.4, 1.0, k=0.5) is False
    assert should_pull_for_lead_move("BUY", +2.0, 1.0, k=0.5) is False  # favourable for a bid
    assert should_pull_for_lead_move("SELL", +0.6, 1.0, k=0.5) is True
    assert should_pull_for_lead_move("SELL", None, 1.0) is False  # no lead data: never pull
    assert should_pull_for_lead_move("BUY", -0.04, 0.0, k=0.5) is False  # spread floored at 0.1 bp


def test_defensive_maker_emits_and_counts_lead_pulls_separately() -> None:
    algo = DefensiveMakerExecutionAlgorithm()
    cmd = algo.evaluate_order_action(
        instrument_id="BTC", desired_side="BUY", current_mid=100.005, best_bid=100.0, best_ask=100.01,
        resting_order_price=100.0, resting_order_side="BUY", ts_ns=1, lead_mid_change_bps=-3.0,
    )
    assert cmd.action == ExecutionAction.CANCEL_LEAD_VENUE_PULL and "LEAD_VENUE_PULL" in cmd.reason
    assert algo.lead_pull_count("BTC") == 1
    calm = algo.evaluate_order_action(
        instrument_id="BTC", desired_side="BUY", current_mid=100.005, best_bid=100.0, best_ask=100.01,
        resting_order_price=100.0, resting_order_side="BUY", ts_ns=2, lead_mid_change_bps=-0.1,
    )
    assert calm.action != ExecutionAction.CANCEL_LEAD_VENUE_PULL and algo.lead_pull_count("BTC") == 1
    shield = algo.evaluate_order_action(
        instrument_id="BTC", desired_side="BUY", current_mid=100.005, best_bid=100.0, best_ask=100.01,
        resting_order_price=100.0, resting_order_side="BUY", ts_ns=3, vpin=0.9, lead_mid_change_bps=-3.0,
    )
    assert shield.action == ExecutionAction.CANCEL_ALL_TOXIC_SHIELD and algo.lead_pull_count("BTC") == 1  # toxic shield first, counted apart


def _ev(t: int, src: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {"recv_ms": t, "src": src, "raw": raw}


def _bbo(t: int, bid: float, ask: float) -> dict[str, Any]:
    return _ev(t, "hl", {"channel": "bbo", "data": {"coin": "BTC", "time": t, "bbo": [{"px": str(bid), "sz": "1", "n": 1}, {"px": str(ask), "sz": "1", "n": 1}]}})


def _binance(t: int, mid: float) -> dict[str, Any]:
    return _ev(t, "binance", {"stream": "btcusdt@bookTicker", "data": {"e": "bookTicker", "s": "BTCUSDT", "b": str(mid - 0.5), "a": str(mid + 0.5), "T": t}})


def _trade(t: int, side: str, px: float) -> dict[str, Any]:
    return _ev(t, "hl", {"channel": "trades", "data": [{"coin": "BTC", "side": side, "px": str(px), "sz": "0.1", "time": t}]})


def _lead_then_hl_drop() -> list[dict[str, Any]]:
    ev = [_bbo(0, 100_000.0, 100_001.0), _binance(0, 100_000.5)]
    ev += [_binance(t, 100_000.5) for t in range(100, 1000, 100)]
    ev.append(_binance(1_000, 99_970.0))  # the lead venue drops 3 bps
    ev.append(_trade(1_300, "A", 100_000.0))  # 300 ms later a seller hits the HL bid
    ev.append(_bbo(1_400, 99_970.0, 99_971.0))  # HL catches up
    ev += [_bbo(t, 99_970.0, 99_971.0) for t in range(2_000, 40_000, 1_000)]
    return ev


def test_simulator_without_pull_takes_the_adverse_fill() -> None:
    res = simulate_quotes(_lead_then_hl_drop(), "BTC", QuoteSimConfig(lead_pull=False))
    assert len(res.fills) == 1 and res.fills[0].side == "BUY" and res.fills[0].price == 100_000.0
    assert res.fills[0].adverse and res.adverse_fills == 1
    assert res.fills[0].markouts_bps[5_000] is not None and res.fills[0].markouts_bps[5_000] < -2.5


def test_simulator_with_pull_cancels_before_the_toxic_print() -> None:
    res = simulate_quotes(_lead_then_hl_drop(), "BTC", QuoteSimConfig(lead_pull=True, k=0.5))
    assert res.pulls >= 1
    assert all(f.price != 100_000.0 for f in res.fills)  # the stale bid was gone when the seller arrived
    assert res.adverse_fills == 0
