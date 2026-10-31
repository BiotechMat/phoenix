"""Queue-position quote simulator (REINFORCEMENT_PLAN task P4.3).

A posted quote joins the back of the displayed size at Hyperliquid's best price. Trades at our price use up
the queue ahead of us; size drops not explained by trades are cancels, assumed pro-rata ahead of and behind
us. The quote fills when the queue ahead is exhausted by a trade at our price, or when the best price trades
through it. Two policies:

- ``repeg`` (current): re-peg to the new best after a cooldown whenever the best moves, losing queue place.
- ``queue_aware`` (plan): keep the order while it sits in the front third of its queue, unless the
  lead-venue pull fires; after a fill wait ``post_fill_cooldown_ms`` before quoting that side again.

Net PnL per fill is the 30-second mid-to-mid markout minus the maker fee. The maker/taker switch rule
(``prefer_taker``) is evaluated for each fill and reported: on Hyperliquid majors the half-spread is a
fraction of a basis point against a 4.5 bp taker fee, so it essentially never prefers taking.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import bisect
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.execution.defensive_maker import should_pull_for_lead_move
from core.streams import BybitTopOfBook, LeadMidBuffer, parse_binance_book_ticker, parse_hyperliquid_trades


def prefer_taker(half_spread_bps: float, taker_fee_bps: float, expected_adverse_bps: float) -> bool:
    """Cross the spread when waiting is expected to cost more than paying it: (half-spread - fee) > adverse move."""
    return (half_spread_bps - taker_fee_bps) > expected_adverse_bps


class QueueSimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy: Literal["repeg", "queue_aware"] = "repeg"
    lead_pull: bool = False
    k: float = Field(default=0.5, gt=0.0)
    window_ms: int = Field(default=700, ge=1)
    repeg_cooldown_ms: int = Field(default=500, ge=0)
    front_fraction: float = Field(default=1.0 / 3.0, gt=0.0, le=1.0)
    post_fill_cooldown_ms: int = Field(default=0, ge=0)
    repost_after_pull_ms: int = Field(default=1_000, ge=0)
    maker_fee_bps: float = Field(default=1.5, ge=0.0)
    taker_fee_bps: float = Field(default=4.5, ge=0.0)


class QueueFill(BaseModel):
    model_config = ConfigDict(frozen=True)

    side: Literal["BUY", "SELL"]
    price: float
    ts_ms: int
    markout_30s_bps: float | None
    net_bps: float | None
    half_spread_bps: float
    taker_preferred: bool


class QueueSimResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: QueueSimConfig
    fills: list[QueueFill]
    repegs: int
    pulls: int
    holds_in_front: int

    @property
    def net_per_fill_bps(self) -> float | None:
        vals = [f.net_bps for f in self.fills if f.net_bps is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def total_net_bps(self) -> float:
        return sum(f.net_bps for f in self.fills if f.net_bps is not None)


class _Quote:
    def __init__(self, price: float, ahead: float, displayed: float, now: int) -> None:
        self.price = price
        self.ahead = ahead
        self.displayed = displayed
        self.traded_since = 0.0
        self.posted = now
        self.last_mod = now


def simulate_queue(events: Iterable[Mapping[str, Any]], coin: str, config: QueueSimConfig) -> QueueSimResult:
    coin = coin.upper()
    lead = LeadMidBuffer(retention_ms=max(2_000, config.window_ms * 3))
    bybit = BybitTopOfBook()
    best = {"BUY": 0.0, "SELL": 0.0}
    size = {"BUY": 0.0, "SELL": 0.0}
    quotes: dict[str, _Quote | None] = {"BUY": None, "SELL": None}
    wait_until = {"BUY": 0, "SELL": 0}
    raw_fills: list[tuple[str, float, int, float]] = []
    mid_ts: list[int] = []
    mid_vals: list[float] = []
    repegs = pulls = holds = 0

    def fill(side: str, now: int) -> None:
        q = quotes[side]
        assert q is not None
        half = (best["SELL"] - best["BUY"]) / ((best["SELL"] + best["BUY"]) / 2.0) * 10_000.0 / 2.0 if best["BUY"] > 0 else 0.0
        raw_fills.append((side, q.price, now, half))
        quotes[side] = None
        wait_until[side] = now + config.post_fill_cooldown_ms

    for ev in events:
        now = int(ev["recv_ms"])
        raw = ev.get("raw") or {}
        src = ev.get("src")
        if src == "binance":
            p = parse_binance_book_ticker(raw)
            if p is not None and p[0] == coin:
                lead.add("binance", coin, p[1], p[2])
            continue
        if src == "bybit":
            p2 = bybit.update(raw)
            if p2 is not None and p2[0] == coin:
                lead.add("bybit", coin, p2[1], p2[2])
            continue
        if src != "hl":
            continue
        if raw.get("channel") == "trades":
            for c, tr in parse_hyperliquid_trades(raw):
                if c != coin:
                    continue
                side = "BUY" if tr.side < 0 else "SELL"  # a taker sell hits resting bids
                q = quotes[side]
                if q is None:
                    continue
                at_or_through = tr.price <= q.price if side == "BUY" else tr.price >= q.price
                if not at_or_through:
                    continue
                if (side == "BUY" and tr.price < q.price) or (side == "SELL" and tr.price > q.price):
                    fill(side, now)  # traded through our level
                    continue
                q.traded_since += tr.size
                q.ahead -= tr.size
                if q.ahead < 0.0:
                    fill(side, now)
            continue
        if raw.get("channel") != "bbo":
            continue
        data = raw.get("data") or {}
        lv = data.get("bbo") or []
        if str(data.get("coin", "")).upper() != coin or len(lv) != 2 or not lv[0] or not lv[1]:
            continue
        best["BUY"], best["SELL"] = float(lv[0]["px"]), float(lv[1]["px"])
        size["BUY"], size["SELL"] = float(lv[0]["sz"]), float(lv[1]["sz"])
        mid = (best["BUY"] + best["SELL"]) / 2.0
        mid_ts.append(now)
        mid_vals.append(mid)
        spread_bps = (best["SELL"] - best["BUY"]) / mid * 10_000.0
        move = lead.lead_mid_change_bps(coin, config.window_ms) if config.lead_pull else None
        for side in ("BUY", "SELL"):
            q = quotes[side]
            b = best[side]
            if q is not None:
                through = b < q.price if side == "BUY" else b > q.price
                if through:
                    fill(side, now)
                    continue
                if b == q.price:
                    # Cancels ahead of us: displayed drop not explained by trades, pro-rata.
                    expected = q.displayed - q.traded_since
                    cancels = max(0.0, expected - size[side])
                    if expected > 0:
                        q.ahead = max(0.0, q.ahead - cancels * (q.ahead / expected))
                    q.displayed = size[side]
                    q.traded_since = 0.0
                if config.lead_pull and should_pull_for_lead_move(side, move, spread_bps, config.k):  # type: ignore[arg-type]
                    quotes[side] = None
                    wait_until[side] = now + config.repost_after_pull_ms
                    pulls += 1
                    continue
                if b != q.price and now - q.last_mod >= config.repeg_cooldown_ms:
                    in_front = q.displayed > 0 and q.ahead / max(q.displayed, 1e-12) <= config.front_fraction
                    if config.policy == "queue_aware" and in_front:
                        holds += 1
                        continue
                    quotes[side] = _Quote(b, size[side], size[side], now)
                    repegs += 1
            elif now >= wait_until[side] and b > 0:
                quotes[side] = _Quote(b, size[side], size[side], now)

    def mid_at(t: int) -> float | None:
        i = bisect.bisect_right(mid_ts, t) - 1
        if i < 0 or not mid_ts or t > mid_ts[-1]:
            return None
        return mid_vals[i]

    out: list[QueueFill] = []
    for side, px, t, half in raw_fills:
        m = mid_at(t + 30_000)
        mk = None if m is None else (1.0 if side == "BUY" else -1.0) * (m - px) / px * 10_000.0
        out.append(
            QueueFill(
                side=side, price=px, ts_ms=t, markout_30s_bps=mk, net_bps=None if mk is None else mk - config.maker_fee_bps,  # type: ignore[arg-type]
                half_spread_bps=half, taker_preferred=prefer_taker(half, config.taker_fee_bps, max(0.0, -(mk or 0.0))),
            )
        )
    return QueueSimResult(config=config, fills=out, repegs=repegs, pulls=pulls, holds_in_front=holds)


__all__ = ["QueueFill", "QueueSimConfig", "QueueSimResult", "prefer_taker", "simulate_queue"]
