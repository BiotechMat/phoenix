"""P4.3 tests: queue-position fill model, cancels, trade-through, front-of-queue holding, taker rule."""

from __future__ import annotations

from typing import Any

from research.queue_sim import QueueSimConfig, prefer_taker, simulate_queue


def _bbo(t: int, bid: float, bsz: float, ask: float, asz: float) -> dict[str, Any]:
    return {"recv_ms": t, "src": "hl", "raw": {"channel": "bbo", "data": {"coin": "BTC", "time": t, "bbo": [{"px": str(bid), "sz": str(bsz), "n": 1}, {"px": str(ask), "sz": str(asz), "n": 1}]}}}


def _sell(t: int, px: float, sz: float) -> dict[str, Any]:
    return {"recv_ms": t, "src": "hl", "raw": {"channel": "trades", "data": [{"coin": "BTC", "side": "A", "px": str(px), "sz": str(sz), "time": t}]}}


def _tail(t0: int, bid: float = 100.0) -> list[dict[str, Any]]:
    return [_bbo(t, bid, 5.0, bid + 0.01, 5.0) for t in range(t0, t0 + 40_000, 1_000)]


def test_fill_only_after_queue_ahead_is_consumed() -> None:
    ev = [_bbo(0, 100.0, 5.0, 100.01, 5.0), _sell(100, 100.0, 3.0), _sell(200, 100.0, 1.9)]
    res = simulate_queue(ev + _tail(1_000), "BTC", QueueSimConfig())
    assert res.fills == []  # 4.9 of the 5 ahead traded: still waiting
    res2 = simulate_queue(ev + [_sell(300, 100.0, 0.2)] + _tail(1_000), "BTC", QueueSimConfig())
    assert len(res2.fills) == 1 and res2.fills[0].price == 100.0 and res2.fills[0].ts_ms == 300


def test_cancels_ahead_move_us_up_pro_rata() -> None:
    ev = [_bbo(0, 100.0, 5.0, 100.01, 5.0), _bbo(100, 100.0, 1.0, 100.01, 5.0), _sell(200, 100.0, 1.5)]
    res = simulate_queue(ev + _tail(1_000), "BTC", QueueSimConfig())
    assert len(res.fills) == 1  # 4 cancelled ahead of us left 1 ahead; a 1.5 print reaches us


def test_trade_through_fills_at_our_price() -> None:
    ev = [_bbo(0, 100.0, 5.0, 100.01, 5.0), _bbo(100, 99.9, 2.0, 99.91, 2.0)]
    res = simulate_queue(ev + _tail(1_000, bid=99.9), "BTC", QueueSimConfig())
    assert [f.price for f in res.fills if f.side == "BUY"] == [100.0]
    assert res.fills[0].markout_30s_bps is not None and res.fills[0].markout_30s_bps < 0  # filled into a falling market


def test_queue_aware_policy_holds_a_front_of_queue_order_that_repeg_gives_up() -> None:
    ev = [_bbo(0, 100.0, 3.0, 100.02, 5.0), _sell(100, 100.0, 2.5)]  # 0.5 left ahead of us: front sixth
    ev += [_bbo(t, 100.01, 4.0, 100.02, 5.0) for t in range(1_000, 5_000, 1_000)]  # a better bid appears
    repeg = simulate_queue(ev, "BTC", QueueSimConfig(policy="repeg"))
    aware = simulate_queue(ev, "BTC", QueueSimConfig(policy="queue_aware"))
    assert repeg.repegs >= 1 and aware.holds_in_front >= 1
    assert aware.repegs < repeg.repegs


def test_taker_rule() -> None:
    assert prefer_taker(half_spread_bps=0.05, taker_fee_bps=4.5, expected_adverse_bps=0.0) is False
    assert prefer_taker(half_spread_bps=12.0, taker_fee_bps=4.5, expected_adverse_bps=3.0) is True
