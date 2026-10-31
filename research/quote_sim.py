"""Resting-quote simulator for lead-venue pulls and markouts (REINFORCEMENT_PLAN P4.1 acceptance, P4.2 fixture).

Replays a recording of Hyperliquid best bid/offer and trades plus Binance/Bybit top of book. One unit rests
at each side of Hyperliquid's best price and re-pegs to the best after a cooldown. A resting bid fills
when a Hyperliquid taker sell prints at or through its price (and the mirror for asks); queue position is
ignored, which is optimistic but identical across policies. With the lead pull enabled, a quote is
cancelled when the lead-venue mid (Binance/Bybit median) moved against it by more than ``k`` x the HL
spread within ``window_ms``, and re-posted only after ``repost_after_pull_ms``.

Every fill is scored by its mid-to-mid markout at 1 s, 5 s, 10 s and 30 s; a fill is "adverse" when its
5-second markout is negative. Events are processed in local receive order, the clock a live quoter sees.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import bisect
import gzip
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.execution.defensive_maker import should_pull_for_lead_move
from core.streams import BybitTopOfBook, LeadMidBuffer, parse_binance_book_ticker, parse_hyperliquid_trades

MARKOUT_HORIZONS_MS: tuple[int, ...] = (1_000, 5_000, 10_000, 30_000)


class QuoteSimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    lead_pull: bool = False
    k: float = Field(default=0.5, gt=0.0)
    window_ms: int = Field(default=700, ge=1)
    repeg_cooldown_ms: int = Field(default=500, ge=0)
    repost_after_fill_ms: int = Field(default=500, ge=0)
    repost_after_pull_ms: int = Field(default=1_000, ge=0)


class SimFill(BaseModel):
    model_config = ConfigDict(frozen=True)

    coin: str
    side: Literal["BUY", "SELL"]
    price: float
    ts_ms: int
    markouts_bps: dict[int, float | None]

    @property
    def adverse(self) -> bool:
        m = self.markouts_bps.get(5_000)
        return m is not None and m < 0.0


class SimResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: QuoteSimConfig
    fills: list[SimFill]
    pulls: int
    duration_ms: int

    @property
    def adverse_fills(self) -> int:
        return sum(1 for f in self.fills if f.adverse)

    @property
    def adverse_rate(self) -> float:
        return self.adverse_fills / len(self.fills) if self.fills else 0.0

    def median_markout(self, horizon_ms: int = 5_000) -> float | None:
        vals = sorted(v for f in self.fills if (v := f.markouts_bps.get(horizon_ms)) is not None)
        if not vals:
            return None
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def load_recording(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    events.sort(key=lambda e: int(e["recv_ms"]))
    return events


def _hl_bbo(raw: Mapping[str, Any]) -> tuple[str, float, float] | None:
    if raw.get("channel") != "bbo":
        return None
    data = raw.get("data") or {}
    levels = data.get("bbo") or []
    if len(levels) != 2 or levels[0] is None or levels[1] is None:
        return None
    try:
        return str(data["coin"]).upper(), float(levels[0]["px"]), float(levels[1]["px"])
    except (KeyError, TypeError, ValueError):
        return None


def simulate_quotes(events: Iterable[Mapping[str, Any]], coin: str, config: QuoteSimConfig) -> SimResult:
    coin = coin.upper()
    lead = LeadMidBuffer(retention_ms=max(2_000, config.window_ms * 3))
    bybit_book = BybitTopOfBook()
    bid_px = ask_px = 0.0
    quotes: dict[str, float | None] = {"BUY": None, "SELL": None}
    last_mod: dict[str, int] = {"BUY": -10**12, "SELL": -10**12}
    repost_at: dict[str, int] = {"BUY": 0, "SELL": 0}
    fills: list[tuple[str, float, int]] = []
    mid_ts: list[int] = []
    mid_vals: list[float] = []
    pulls = 0
    first = last = None

    for ev in events:
        now = int(ev["recv_ms"])
        first = now if first is None else first
        last = now
        raw = ev.get("raw") or {}
        src = ev.get("src")
        if src == "binance":
            parsed = parse_binance_book_ticker(raw)
            if parsed is not None and parsed[0] == coin:
                lead.add("binance", coin, parsed[1], parsed[2])
        elif src == "bybit":
            parsed2 = bybit_book.update(raw)
            if parsed2 is not None and parsed2[0] == coin:
                lead.add("bybit", coin, parsed2[1], parsed2[2])
        elif src == "hl":
            bbo = _hl_bbo(raw)
            if bbo is not None and bbo[0] == coin:
                bid_px, ask_px = bbo[1], bbo[2]
                mid_ts.append(now)
                mid_vals.append((bid_px + ask_px) / 2.0)
            for c, trade in parse_hyperliquid_trades(raw):
                if c != coin:
                    continue
                if trade.side < 0 and quotes["BUY"] is not None and trade.price <= float(quotes["BUY"] or 0.0):
                    fills.append(("BUY", float(quotes["BUY"] or 0.0), now))
                    quotes["BUY"] = None
                    repost_at["BUY"] = now + config.repost_after_fill_ms
                if trade.side > 0 and quotes["SELL"] is not None and trade.price >= float(quotes["SELL"] or 0.0):
                    fills.append(("SELL", float(quotes["SELL"] or 0.0), now))
                    quotes["SELL"] = None
                    repost_at["SELL"] = now + config.repost_after_fill_ms
        if bid_px <= 0.0 or ask_px <= 0.0:
            continue
        spread_bps = (ask_px - bid_px) / ((ask_px + bid_px) / 2.0) * 10_000.0
        if config.lead_pull:
            move = lead.lead_mid_change_bps(coin, config.window_ms)
            for side in ("BUY", "SELL"):
                if quotes[side] is not None and should_pull_for_lead_move(side, move, spread_bps, config.k):
                    quotes[side] = None
                    repost_at[side] = now + config.repost_after_pull_ms
                    pulls += 1
        for side, best in (("BUY", bid_px), ("SELL", ask_px)):
            if quotes[side] is None:
                if now >= repost_at[side]:
                    quotes[side] = best
                    last_mod[side] = now
            elif quotes[side] != best and now - last_mod[side] >= config.repeg_cooldown_ms:
                quotes[side] = best
                last_mod[side] = now

    def mid_at(t: int) -> float | None:
        i = bisect.bisect_right(mid_ts, t) - 1
        if i < 0 or not mid_ts or t > mid_ts[-1]:
            return None
        return mid_vals[i]

    sim_fills: list[SimFill] = []
    for side, px, t in fills:
        sign = 1.0 if side == "BUY" else -1.0
        marks: dict[int, float | None] = {}
        for h in MARKOUT_HORIZONS_MS:
            m = mid_at(t + h)
            marks[h] = None if m is None else sign * (m - px) / px * 10_000.0
        sim_fills.append(SimFill(coin=coin, side=side, price=px, ts_ms=t, markouts_bps=marks))  # type: ignore[arg-type]
    return SimResult(config=config, fills=sim_fills, pulls=pulls, duration_ms=(last or 0) - (first or 0))


def compare_policies(events: Sequence[Mapping[str, Any]], coins: Sequence[str] = ("BTC", "ETH", "SOL"), k: float = 0.5) -> dict[str, tuple[SimResult, SimResult]]:
    return {
        c: (simulate_quotes(events, c, QuoteSimConfig(lead_pull=False)), simulate_quotes(events, c, QuoteSimConfig(lead_pull=True, k=k)))
        for c in coins
    }


__all__ = ["MARKOUT_HORIZONS_MS", "QuoteSimConfig", "SimFill", "SimResult", "compare_policies", "load_recording", "simulate_quotes"]
