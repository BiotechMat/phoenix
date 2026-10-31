"""Streaming feeds: trades, liquidations and lead-venue mids (REINFORCEMENT_PLAN P1.1, P1.3, P1.5).

- Trades: Hyperliquid websocket ``trades`` per coin. Kept in a thread-safe ``TradeTape`` for 25 hours,
  flushed to Parquet part files, and used for (a) real taker buy/sell dollar volume per bar and
  (b) the quarter-hour boundary imbalance feature.
- Liquidations: Bybit websocket ``allLiquidation.{SYMBOL}`` (primary) with OKX REST
  ``public/liquidation-orders`` polling as the fallback. Aggregated per bar by side.
- Lead mids: Binance futures ``bookTicker`` and Bybit ``orderbook.1`` top of book with exchange
  timestamps, kept for a short window; ``lead_mid_change_bps(window_ms)`` reports the move.

Coverage is tracked per stream: a bar's taker flow or liquidation volume is used only when the stream
was connected for the whole bar. Otherwise the field stays unavailable; nothing is filled in.

Parsers and buffers are pure or lock-protected and are tested on recorded payloads. ``StreamHub``
runs the websockets on an asyncio loop in a daemon thread with reconnect and backoff.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
import json
import logging
import os
from pathlib import Path
import ssl
import threading
import time
from typing import Any, Final, Literal
import urllib.request

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("phoenix.streams")

HYPERLIQUID_WS_URL: Final[str] = "wss://api.hyperliquid.xyz/ws"
BINANCE_FUTURES_WS_URL: Final[str] = "wss://fstream.binance.com/stream"
BYBIT_LINEAR_WS_URL: Final[str] = "wss://stream.bybit.com/v5/public/linear"
OKX_REST_URL: Final[str] = "https://www.okx.com/api/v5"

QUARTER_HOUR_MS: Final[int] = 900_000
DAY_MS: Final[int] = 86_400_000
TAPE_RETENTION_MS: Final[int] = 25 * 3_600_000
LEAD_MID_RETENTION_MS: Final[int] = 2_000
BOUNDARY_WINDOWS_S: Final[tuple[int, ...]] = (10, 30, 60)
DEFAULT_MAX_SILENCE_MS: Final[int] = 120_000

Side = Literal[1, -1]


# ================================================================================================
# Records
# ================================================================================================


class Trade(BaseModel):
    """One aggressor trade. ``side`` is +1 for a taker buy and -1 for a taker sell."""

    model_config = ConfigDict(frozen=True)

    ts_ms: int = Field(ge=0)
    price: float = Field(gt=0.0)
    size: float = Field(ge=0.0)
    side: Side

    @property
    def notional_usd(self) -> float:
        return self.price * self.size


class LiquidationEvent(BaseModel):
    """A forced close. ``position_side`` is the side of the position that was liquidated."""

    model_config = ConfigDict(frozen=True)

    ts_ms: int = Field(ge=0)
    coin: str
    position_side: Literal["long", "short"]
    size_base: float = Field(ge=0.0)
    price: float = Field(gt=0.0)
    venue: str

    @property
    def notional_usd(self) -> float:
        return self.size_base * self.price


class BoundaryImbalance(BaseModel):
    """Signed taker volume after a quarter-hour boundary, raw and normalised by trailing per-window volume."""

    model_config = ConfigDict(frozen=True)

    coin: str
    boundary_ms: int
    signed_volume: dict[int, float]
    total_volume: dict[int, float]
    normalized: dict[int, float | None]
    trailing_hours_used: float = Field(ge=0.0)
    n_trades: int = Field(ge=0)


class BarFlow(BaseModel):
    """Stream-derived fields for one bar; ``None`` when the stream did not cover the whole bar."""

    model_config = ConfigDict(frozen=True)

    coin: str
    start_ms: int
    end_ms: int
    taker_buy_usd: float | None = None
    taker_sell_usd: float | None = None
    n_trades: int = 0
    liquidation_usd: float | None = None
    long_liquidation_usd: float | None = None
    short_liquidation_usd: float | None = None


# ================================================================================================
# Parsers
# ================================================================================================


def _coin_from_usdt_symbol(symbol: str) -> str:
    s = symbol.upper()
    return s[:-4] if s.endswith("USDT") else s


def parse_hyperliquid_trades(msg: Mapping[str, Any]) -> list[tuple[str, Trade]]:
    """``{"channel": "trades", "data": [{coin, side "B"|"A", px, sz, time}, ...]}`` -> ``[(coin, Trade)]``.

    Hyperliquid reports the aggressor side: "B" is a taker buy, "A" a taker sell.
    """
    if msg.get("channel") != "trades":
        return []
    out: list[tuple[str, Trade]] = []
    for row in msg.get("data", []) or []:
        if not isinstance(row, dict):
            continue
        try:
            side: Side = 1 if str(row["side"]).upper() == "B" else -1
            out.append(
                (str(row["coin"]).upper(), Trade(ts_ms=int(row["time"]), price=float(row["px"]), size=float(row["sz"]), side=side))
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_binance_book_ticker(msg: Mapping[str, Any]) -> tuple[str, int, float] | None:
    """Binance futures bookTicker (raw or combined-stream wrapped) -> ``(coin, exchange_ts_ms, mid)``."""
    data = msg.get("data", msg) if isinstance(msg.get("data"), dict) else msg
    if data.get("e") != "bookTicker":
        return None
    try:
        bid, ask = float(data["b"]), float(data["a"])
        ts = int(data.get("T") or data.get("E") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if bid <= 0.0 or ask <= 0.0:
        return None
    return _coin_from_usdt_symbol(str(data["s"])), ts, (bid + ask) / 2.0


class BybitTopOfBook:
    """Merges Bybit ``orderbook.1`` snapshots and deltas into a mid per symbol."""

    def __init__(self) -> None:
        self._best: dict[str, tuple[float | None, float | None]] = {}

    def update(self, msg: Mapping[str, Any]) -> tuple[str, int, float] | None:
        topic = str(msg.get("topic", ""))
        if not topic.startswith("orderbook.1."):
            return None
        data = msg.get("data") or {}
        if not isinstance(data, dict):
            return None
        symbol = str(data.get("s") or topic.split(".")[-1])
        bid, ask = self._best.get(symbol, (None, None))
        if msg.get("type") == "snapshot":
            bid, ask = None, None
        for px, qty in data.get("b", []) or []:
            bid = float(px) if float(qty) > 0.0 else None
        for px, qty in data.get("a", []) or []:
            ask = float(px) if float(qty) > 0.0 else None
        self._best[symbol] = (bid, ask)
        if bid is None or ask is None:
            return None
        ts = int(msg.get("cts") or msg.get("ts") or 0)
        return _coin_from_usdt_symbol(symbol), ts, (bid + ask) / 2.0


def parse_bybit_all_liquidation(msg: Mapping[str, Any]) -> list[LiquidationEvent]:
    """Bybit ``allLiquidation.{SYMBOL}``: ``data = [{T, s, S, v, p}]``.

    Per Bybit's documentation ``S = "Buy"`` means a long position was liquidated and ``"Sell"`` a short.
    """
    if not str(msg.get("topic", "")).startswith("allLiquidation."):
        return []
    rows = msg.get("data") or []
    if isinstance(rows, dict):
        rows = [rows]
    out: list[LiquidationEvent] = []
    for r in rows:
        try:
            out.append(
                LiquidationEvent(
                    ts_ms=int(r["T"]),
                    coin=_coin_from_usdt_symbol(str(r["s"])),
                    position_side="long" if str(r["S"]).lower() == "buy" else "short",
                    size_base=float(r["v"]),
                    price=float(r["p"]),
                    venue="bybit",
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_okx_liquidation_orders(payload: Mapping[str, Any], contract_value: Mapping[str, float]) -> list[LiquidationEvent]:
    """OKX ``public/liquidation-orders``: sizes are in contracts and converted with the instrument's ``ctVal``."""
    out: list[LiquidationEvent] = []
    for group in payload.get("data", []) or []:
        inst_id = str(group.get("instId", ""))
        coin = inst_id.split("-")[0].upper()
        ct_val = contract_value.get(inst_id)
        if ct_val is None:
            continue
        for d in group.get("details", []) or []:
            try:
                out.append(
                    LiquidationEvent(
                        ts_ms=int(d.get("ts") or d.get("time")),
                        coin=coin,
                        position_side="long" if str(d["posSide"]).lower() == "long" else "short",
                        size_base=float(d["sz"]) * float(ct_val),
                        price=float(d["bkPx"]),
                        venue="okx",
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    out.sort(key=lambda e: e.ts_ms)
    return out


# ================================================================================================
# Coverage
# ================================================================================================


class Coverage:
    """Intervals during which a stream was connected and receiving.

    ``alive(ts)`` extends the open interval; a silence longer than ``max_silence_ms`` or an explicit
    ``break_()`` closes it, so the next ``alive`` opens a new one.
    """

    def __init__(self, max_silence_ms: int = DEFAULT_MAX_SILENCE_MS) -> None:
        self._intervals: list[list[int]] = []
        self._open = False
        self._max_silence = max_silence_ms

    def alive(self, ts_ms: int) -> None:
        if self._open and self._intervals and ts_ms - self._intervals[-1][1] <= self._max_silence:
            self._intervals[-1][1] = max(self._intervals[-1][1], ts_ms)
            return
        self._intervals.append([ts_ms, ts_ms])
        self._open = True

    def break_(self) -> None:
        self._open = False

    def covers(self, start_ms: int, end_ms: int) -> bool:
        return any(a <= start_ms and b >= end_ms for a, b in self._intervals)

    def covered_ms_within(self, start_ms: int, end_ms: int) -> int:
        total = 0
        for a, b in self._intervals:
            lo, hi = max(a, start_ms), min(b, end_ms)
            if hi > lo:
                total += hi - lo
        return total

    def prune(self, before_ms: int) -> None:
        self._intervals = [iv for iv in self._intervals if iv[1] >= before_ms]

    @property
    def last_alive_ms(self) -> int | None:
        return self._intervals[-1][1] if self._intervals else None


# ================================================================================================
# Buffers
# ================================================================================================


class TradeTape:
    """Thread-safe per-coin trade buffer with coverage tracking and Parquet flushing."""

    def __init__(self, coins: Sequence[str], retention_ms: int = TAPE_RETENTION_MS, max_silence_ms: int = DEFAULT_MAX_SILENCE_MS) -> None:
        self._lock = threading.Lock()
        self._coins = tuple(c.upper() for c in coins)
        self._trades: dict[str, deque[Trade]] = {c: deque() for c in self._coins}
        self._coverage: dict[str, Coverage] = {c: Coverage(max_silence_ms) for c in self._coins}
        self._retention = retention_ms
        self._unflushed: dict[str, list[Trade]] = {c: [] for c in self._coins}

    @property
    def coins(self) -> tuple[str, ...]:
        return self._coins

    def add(self, coin: str, trades: Iterable[Trade], received_ms: int | None = None) -> None:
        c = coin.upper()
        if c not in self._trades:
            return
        with self._lock:
            buf = self._trades[c]
            for t in trades:
                buf.append(t)
                self._unflushed[c].append(t)
                self._coverage[c].alive(t.ts_ms)
            if received_ms is not None:
                self._coverage[c].alive(received_ms)
            cutoff = (buf[-1].ts_ms if buf else (received_ms or 0)) - self._retention
            while buf and buf[0].ts_ms < cutoff:
                buf.popleft()
            self._coverage[c].prune(cutoff)

    def heartbeat(self, coin: str, ts_ms: int) -> None:
        with self._lock:
            cov = self._coverage.get(coin.upper())
            if cov is not None:
                cov.alive(ts_ms)

    def break_coverage(self, coin: str | None = None) -> None:
        with self._lock:
            for c in ([coin.upper()] if coin else list(self._coins)):
                if c in self._coverage:
                    self._coverage[c].break_()

    def covers(self, coin: str, start_ms: int, end_ms: int) -> bool:
        with self._lock:
            cov = self._coverage.get(coin.upper())
            return cov.covers(start_ms, end_ms) if cov is not None else False

    def covered_ms_within(self, coin: str, start_ms: int, end_ms: int) -> int:
        with self._lock:
            cov = self._coverage.get(coin.upper())
            return cov.covered_ms_within(start_ms, end_ms) if cov is not None else 0

    def last_alive_ms(self, coin: str) -> int | None:
        with self._lock:
            cov = self._coverage.get(coin.upper())
            return cov.last_alive_ms if cov is not None else None

    def trades_between(self, coin: str, start_ms: int, end_ms: int) -> list[Trade]:
        with self._lock:
            return [t for t in self._trades.get(coin.upper(), deque()) if start_ms <= t.ts_ms < end_ms]

    def taker_volumes(self, coin: str, start_ms: int, end_ms: int) -> tuple[float, float, int]:
        """``(buy_usd, sell_usd, n_trades)`` over ``[start_ms, end_ms)``."""
        buy = sell = 0.0
        n = 0
        for t in self.trades_between(coin, start_ms, end_ms):
            n += 1
            if t.side > 0:
                buy += t.notional_usd
            else:
                sell += t.notional_usd
        return buy, sell, n

    def flush_parquet(self, directory: Path) -> int:
        """Write trades received since the last flush as one part file per coin; returns rows written."""
        with self._lock:
            pending = {c: list(v) for c, v in self._unflushed.items() if v}
            for c in pending:
                self._unflushed[c] = []
        written = 0
        for coin, trades in pending.items():
            out_dir = directory / coin.lower()
            out_dir.mkdir(parents=True, exist_ok=True)
            df = pl.DataFrame(
                {
                    "ts_ms": [t.ts_ms for t in trades],
                    "price": [t.price for t in trades],
                    "size": [t.size for t in trades],
                    "side": [int(t.side) for t in trades],
                },
                schema={"ts_ms": pl.Int64, "price": pl.Float64, "size": pl.Float64, "side": pl.Int8},
            )
            df.write_parquet(out_dir / f"trades_{trades[0].ts_ms}_{trades[-1].ts_ms}.parquet")
            written += df.height
        return written


class LiquidationBook:
    """Thread-safe liquidation event store with per-coin coverage of the liquidation stream."""

    def __init__(self, coins: Sequence[str], retention_ms: int = TAPE_RETENTION_MS, max_silence_ms: int = DEFAULT_MAX_SILENCE_MS) -> None:
        self._lock = threading.Lock()
        self._coins = tuple(c.upper() for c in coins)
        self._events: dict[str, deque[LiquidationEvent]] = {c: deque() for c in self._coins}
        self._seen: set[tuple[str, int, float, float, str]] = set()
        self._coverage: dict[str, Coverage] = {c: Coverage(max_silence_ms) for c in self._coins}
        self._retention = retention_ms

    def add(self, events: Iterable[LiquidationEvent]) -> int:
        added = 0
        with self._lock:
            for e in events:
                key = (e.coin, e.ts_ms, e.size_base, e.price, e.position_side)
                if e.coin not in self._events or key in self._seen:
                    continue
                self._seen.add(key)
                self._events[e.coin].append(e)
                added += 1
            for c, buf in self._events.items():
                if buf:
                    cutoff = buf[-1].ts_ms - self._retention
                    while buf and buf[0].ts_ms < cutoff:
                        old = buf.popleft()
                        self._seen.discard((old.coin, old.ts_ms, old.size_base, old.price, old.position_side))
        return added

    def heartbeat(self, ts_ms: int, coins: Iterable[str] | None = None) -> None:
        with self._lock:
            for c in (coins or self._coins):
                cov = self._coverage.get(c.upper())
                if cov is not None:
                    cov.alive(ts_ms)

    def break_coverage(self) -> None:
        with self._lock:
            for cov in self._coverage.values():
                cov.break_()

    def covers(self, coin: str, start_ms: int, end_ms: int) -> bool:
        with self._lock:
            cov = self._coverage.get(coin.upper())
            return cov.covers(start_ms, end_ms) if cov is not None else False

    def last_alive_ms(self, coin: str) -> int | None:
        with self._lock:
            cov = self._coverage.get(coin.upper())
            return cov.last_alive_ms if cov is not None else None

    def volume_usd(self, coin: str, start_ms: int, end_ms: int) -> tuple[float, float]:
        """``(long_liquidated_usd, short_liquidated_usd)`` over ``[start_ms, end_ms)``."""
        long_usd = short_usd = 0.0
        with self._lock:
            for e in self._events.get(coin.upper(), deque()):
                if start_ms <= e.ts_ms < end_ms:
                    if e.position_side == "long":
                        long_usd += e.notional_usd
                    else:
                        short_usd += e.notional_usd
        return long_usd, short_usd

    def recent(self, coin: str, n: int = 10) -> list[LiquidationEvent]:
        with self._lock:
            return list(self._events.get(coin.upper(), deque()))[-n:]


class LeadMidBuffer:
    """Recent top-of-book mids per (venue, coin) keyed by exchange timestamp."""

    def __init__(self, retention_ms: int = LEAD_MID_RETENTION_MS) -> None:
        self._lock = threading.Lock()
        self._mids: dict[tuple[str, str], deque[tuple[int, float]]] = {}
        self._retention = retention_ms

    def add(self, venue: str, coin: str, ts_ms: int, mid: float) -> None:
        key = (venue, coin.upper())
        with self._lock:
            buf = self._mids.setdefault(key, deque())
            if buf and ts_ms < buf[-1][0]:
                return  # out-of-order update: ignore
            buf.append((ts_ms, mid))
            cutoff = ts_ms - self._retention
            while len(buf) > 1 and buf[1][0] <= cutoff:
                buf.popleft()

    def venues(self, coin: str) -> list[str]:
        with self._lock:
            return sorted(v for (v, c) in self._mids if c == coin.upper())

    def latest(self, venue: str, coin: str) -> tuple[int, float] | None:
        with self._lock:
            buf = self._mids.get((venue, coin.upper()))
            return buf[-1] if buf else None

    def lead_mid_change_bps(self, coin: str, window_ms: int, venue: str | None = None) -> float | None:
        """Move of the lead-venue mid over the last ``window_ms`` (exchange time), in bps.

        With ``venue=None`` the median across venues that have a full window is returned. ``None`` when no
        venue has an observation at or before the window start.
        """
        moves: list[float] = []
        with self._lock:
            keys = [k for k in self._mids if k[1] == coin.upper() and (venue is None or k[0] == venue)]
            for key in keys:
                buf = self._mids[key]
                if not buf:
                    continue
                last_ts, last_mid = buf[-1]
                start_ts = last_ts - window_ms
                base: float | None = None
                for ts, mid in buf:
                    if ts <= start_ts:
                        base = mid
                    else:
                        break
                if base is not None and base > 0.0:
                    moves.append((last_mid - base) / base * 10_000.0)
        if not moves:
            return None
        moves.sort()
        mid_i = len(moves) // 2
        return moves[mid_i] if len(moves) % 2 else (moves[mid_i - 1] + moves[mid_i]) / 2.0


# ================================================================================================
# Boundary imbalance (P1.1)
# ================================================================================================


def quarter_hour_boundaries(start_ms: int, end_ms: int) -> list[int]:
    """Every :00/:15/:30/:45 boundary in ``[start_ms, end_ms]``."""
    first = ((start_ms + QUARTER_HOUR_MS - 1) // QUARTER_HOUR_MS) * QUARTER_HOUR_MS
    return list(range(first, end_ms + 1, QUARTER_HOUR_MS))


def compute_boundary_imbalance(
    coin: str,
    boundary_ms: int,
    trades_after: Sequence[Trade],
    trailing_volume_base: float,
    trailing_covered_ms: int,
    windows_s: Sequence[int] = BOUNDARY_WINDOWS_S,
) -> BoundaryImbalance:
    """Signed taker volume in the first ``w`` seconds after ``boundary_ms``.

    Normalised by the trailing average per-window volume: ``trailing_volume_base * w / covered_seconds``,
    where the trailing window is up to 24 hours before the boundary (the covered part of it).
    """
    signed: dict[int, float] = {}
    total: dict[int, float] = {}
    norm: dict[int, float | None] = {}
    covered_s = trailing_covered_ms / 1000.0
    n = 0
    for w in windows_s:
        end = boundary_ms + w * 1000
        s = 0.0
        tot = 0.0
        for t in trades_after:
            if boundary_ms <= t.ts_ms < end:
                s += t.side * t.size
                tot += t.size
        signed[w] = s
        total[w] = tot
        avg_per_window = trailing_volume_base * w / covered_s if covered_s > 0.0 else 0.0
        norm[w] = s / avg_per_window if avg_per_window > 0.0 else None
    n = sum(1 for t in trades_after if boundary_ms <= t.ts_ms < boundary_ms + max(windows_s) * 1000)
    return BoundaryImbalance(
        coin=coin.upper(),
        boundary_ms=boundary_ms,
        signed_volume=signed,
        total_volume=total,
        normalized=norm,
        trailing_hours_used=covered_s / 3600.0,
        n_trades=n,
    )


def boundary_imbalance_from_tape(tape: TradeTape, coin: str, boundary_ms: int, windows_s: Sequence[int] = BOUNDARY_WINDOWS_S) -> BoundaryImbalance | None:
    """Compute the feature from the tape when the stream covered the whole post-boundary window."""
    horizon_ms = max(windows_s) * 1000
    if not tape.covers(coin, boundary_ms, boundary_ms + horizon_ms):
        return None
    trailing_start = boundary_ms - DAY_MS
    trailing = tape.trades_between(coin, trailing_start, boundary_ms)
    covered = tape.covered_ms_within(coin, trailing_start, boundary_ms)
    return compute_boundary_imbalance(
        coin,
        boundary_ms,
        tape.trades_between(coin, boundary_ms, boundary_ms + horizon_ms),
        trailing_volume_base=sum(t.size for t in trailing),
        trailing_covered_ms=covered,
        windows_s=windows_s,
    )


class BoundaryImbalanceTracker:
    """Computes the feature for each boundary once its longest window has elapsed."""

    def __init__(self, tape: TradeTape, windows_s: Sequence[int] = BOUNDARY_WINDOWS_S, keep: int = 96 * 2) -> None:
        self._tape = tape
        self._windows = tuple(windows_s)
        self._keep = keep
        self._lock = threading.Lock()
        self._records: dict[str, deque[BoundaryImbalance]] = {c: deque(maxlen=keep) for c in tape.coins}
        self._done: set[tuple[str, int]] = set()

    def update(self, now_ms: int, retry_ms: int = 300_000) -> list[BoundaryImbalance]:
        """Compute any boundary whose window has elapsed.

        A boundary the tape does not yet cover is retried for ``retry_ms`` after its window ends (quiet
        coins extend coverage only on the next trade or heartbeat), then abandoned.
        """
        horizon = max(self._windows) * 1000
        new: list[BoundaryImbalance] = []
        for boundary in quarter_hour_boundaries(now_ms - 2 * QUARTER_HOUR_MS, now_ms - horizon):
            for coin in self._tape.coins:
                if (coin, boundary) in self._done:
                    continue
                rec = boundary_imbalance_from_tape(self._tape, coin, boundary, self._windows)
                if rec is None:
                    if now_ms - (boundary + horizon) >= retry_ms:
                        self._done.add((coin, boundary))
                    continue
                self._done.add((coin, boundary))
                with self._lock:
                    self._records[coin].append(rec)
                new.append(rec)
        cutoff = now_ms - 2 * DAY_MS
        self._done = {k for k in self._done if k[1] >= cutoff}
        return new

    def latest(self, coin: str) -> BoundaryImbalance | None:
        with self._lock:
            buf = self._records.get(coin.upper())
            return buf[-1] if buf else None

    def records(self, coin: str) -> list[BoundaryImbalance]:
        with self._lock:
            return list(self._records.get(coin.upper(), ()))


# ================================================================================================
# Hub
# ================================================================================================


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and Path(cafile).exists():
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def fetch_okx_contract_values(inst_ids: Sequence[str], timeout_sec: float = 5.0) -> dict[str, float]:
    """``ctVal`` per OKX swap instrument (base units per contract)."""
    out: dict[str, float] = {}
    for inst in inst_ids:
        req = urllib.request.Request(f"{OKX_REST_URL}/public/instruments?instType=SWAP&instId={inst}", headers={"User-Agent": "ProjectPhoenix/4.1"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("data", [])
        if rows:
            out[inst] = float(rows[0]["ctVal"])
    return out


def fetch_okx_liquidations(coin: str, contract_value: Mapping[str, float], timeout_sec: float = 5.0) -> list[LiquidationEvent]:
    req = urllib.request.Request(
        f"{OKX_REST_URL}/public/liquidation-orders?instType=SWAP&instFamily={coin.upper()}-USDT&state=filled&limit=100",
        headers={"User-Agent": "ProjectPhoenix/4.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_okx_liquidation_orders(payload, contract_value)


class StreamAvailability(BaseModel):
    model_config = ConfigDict(frozen=True)

    trades: bool = False
    liquidations: bool = False
    lead_mids: bool = False


class StreamHub:
    """Owns the buffers and, once started, the websocket tasks feeding them."""

    def __init__(
        self,
        coins: Sequence[str] = ("BTC", "ETH", "SOL"),
        *,
        trades_dir: Path | None = None,
        flush_interval_s: float = 300.0,
        enable_okx_liquidation_poll: bool = True,
    ) -> None:
        self.coins: tuple[str, ...] = tuple(c.upper() for c in coins)
        self.tape = TradeTape(self.coins)
        self.liquidations = LiquidationBook(self.coins)
        self.lead_mids = LeadMidBuffer()
        self.boundaries = BoundaryImbalanceTracker(self.tape)
        from core.execution.markouts import MidHistory

        self.hl_mids = MidHistory()
        self._trades_dir = trades_dir
        self._flush_interval_s = flush_interval_s
        self._okx_poll = enable_okx_liquidation_poll
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._bybit_liq_connected = False
        self._status: dict[str, str] = {}

    # ---------------------------------------------------------------- queries used by the tick

    def bar_flow(self, coin: str, start_ms: int, end_ms: int) -> BarFlow:
        c = coin.upper()
        flow = BarFlow(coin=c, start_ms=start_ms, end_ms=end_ms)
        if self.tape.covers(c, start_ms, end_ms):
            buy, sell, n = self.tape.taker_volumes(c, start_ms, end_ms)
            flow = flow.model_copy(update={"taker_buy_usd": buy, "taker_sell_usd": sell, "n_trades": n})
        if self.liquidations.covers(c, start_ms, end_ms):
            long_usd, short_usd = self.liquidations.volume_usd(c, start_ms, end_ms)
            flow = flow.model_copy(update={"liquidation_usd": long_usd + short_usd, "long_liquidation_usd": long_usd, "short_liquidation_usd": short_usd})
        return flow

    def availability(self, now_ms: int | None = None) -> StreamAvailability:
        now = int(time.time() * 1000) if now_ms is None else now_ms

        def fresh(ts: int | None, max_age: int) -> bool:
            return ts is not None and now - ts <= max_age

        trades_ok = all(fresh(self.tape.last_alive_ms(c), DEFAULT_MAX_SILENCE_MS) for c in self.coins)
        liq_ok = all(fresh(self.liquidations.last_alive_ms(c), DEFAULT_MAX_SILENCE_MS) for c in self.coins)
        mids_ok = False
        for c in self.coins:
            for v in self.lead_mids.venues(c):
                latest = self.lead_mids.latest(v, c)
                if latest is not None and fresh(latest[0], 10_000):
                    mids_ok = True
        return StreamAvailability(trades=trades_ok, liquidations=liq_ok, lead_mids=mids_ok)

    @property
    def status(self) -> dict[str, str]:
        return dict(self._status)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="phoenix-stream-hub", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._thread = None

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Stream hub stopped with error: %s", exc)
        finally:
            # Cancel and drain everything still scheduled (e.g. websocket keepalive tasks) before closing.
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    async def _main(self) -> None:
        tasks = [
            asyncio.create_task(self._run_ws("hyperliquid", HYPERLIQUID_WS_URL, self._hl_subscribe, self._on_hl, ping=lambda: {"method": "ping"}, ping_s=30.0)),
            asyncio.create_task(self._run_ws("binance", self._binance_url(), None, self._on_binance, ping=None, ping_s=0.0)),
            asyncio.create_task(self._run_ws("bybit", BYBIT_LINEAR_WS_URL, self._bybit_subscribe, self._make_bybit_handler(), ping=lambda: {"op": "ping"}, ping_s=20.0)),
            asyncio.create_task(self._periodic()),
        ]
        while not self._stop.is_set():
            await asyncio.sleep(0.5)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ---------------------------------------------------------------- websocket plumbing

    def _binance_url(self) -> str:
        streams = "/".join(f"{c.lower()}usdt@bookTicker" for c in self.coins)
        return f"{BINANCE_FUTURES_WS_URL}?streams={streams}"

    def _hl_subscribe(self) -> list[dict[str, Any]]:
        subs = [{"method": "subscribe", "subscription": {"type": "trades", "coin": c}} for c in self.coins]
        subs += [{"method": "subscribe", "subscription": {"type": "bbo", "coin": c}} for c in self.coins]
        return subs

    def _bybit_subscribe(self) -> list[dict[str, Any]]:
        args = [f"orderbook.1.{c}USDT" for c in self.coins] + [f"allLiquidation.{c}USDT" for c in self.coins]
        return [{"op": "subscribe", "args": args}]

    async def _run_ws(
        self,
        name: str,
        url: str,
        subscribe: Callable[[], list[dict[str, Any]]] | None,
        handler: Callable[[Mapping[str, Any], int], None],
        *,
        ping: Callable[[], dict[str, Any]] | None,
        ping_s: float,
    ) -> None:
        import websockets

        backoff = 1.0
        ctx = _ssl_context()
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ssl=ctx, open_timeout=10, proxy=True, max_size=2**22) as ws:
                    self._status[name] = "connected"
                    backoff = 1.0
                    for sub in (subscribe() if subscribe else []):
                        await ws.send(json.dumps(sub))
                    last_ping = time.monotonic()
                    while not self._stop.is_set():
                        timeout = max(0.5, ping_s - (time.monotonic() - last_ping)) if ping is not None else 5.0
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except asyncio.TimeoutError:
                            raw = None
                        now_ms = int(time.time() * 1000)
                        if raw is not None:
                            try:
                                msg = json.loads(raw)
                            except (TypeError, ValueError):
                                msg = None
                            if isinstance(msg, dict):
                                handler(msg, now_ms)
                        if ping is not None and time.monotonic() - last_ping >= ping_s:
                            await ws.send(json.dumps(ping()))
                            last_ping = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._status[name] = f"reconnecting ({type(exc).__name__})"
                logger.warning("Stream '%s' disconnected (%s); retrying in %.0fs", name, exc, backoff)
            finally:
                self._on_disconnect(name)
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)

    def _on_disconnect(self, name: str) -> None:
        if name == "hyperliquid":
            self.tape.break_coverage()
        elif name == "bybit":
            self._bybit_liq_connected = False
            self.liquidations.break_coverage()

    # ---------------------------------------------------------------- handlers

    def _on_hl(self, msg: Mapping[str, Any], now_ms: int) -> None:
        if msg.get("channel") == "trades":
            by_coin: dict[str, list[Trade]] = {}
            for coin, trade in parse_hyperliquid_trades(msg):
                by_coin.setdefault(coin, []).append(trade)
            for coin, trades in by_coin.items():
                self.tape.add(coin, trades, received_ms=now_ms)
        elif msg.get("channel") == "bbo":
            data = msg.get("data") or {}
            levels = data.get("bbo") or []
            if len(levels) == 2 and levels[0] and levels[1]:
                try:
                    mid = (float(levels[0]["px"]) + float(levels[1]["px"])) / 2.0
                    self.hl_mids.add(str(data.get("coin", "")).upper(), now_ms, mid)
                except (KeyError, TypeError, ValueError):
                    pass
        elif msg.get("channel") in ("pong", "subscriptionResponse"):
            for c in self.coins:
                self.tape.heartbeat(c, now_ms)

    def _on_binance(self, msg: Mapping[str, Any], now_ms: int) -> None:
        parsed = parse_binance_book_ticker(msg)
        if parsed is not None:
            coin, ts, mid = parsed
            self.lead_mids.add("binance", coin, ts, mid)

    def _make_bybit_handler(self) -> Callable[[Mapping[str, Any], int], None]:
        book = BybitTopOfBook()

        def _handler(msg: Mapping[str, Any], now_ms: int) -> None:
            topic = str(msg.get("topic", ""))
            if topic.startswith("orderbook.1."):
                parsed = book.update(msg)
                if parsed is not None:
                    coin, ts, mid = parsed
                    self.lead_mids.add("bybit", coin, ts, mid)
            elif topic.startswith("allLiquidation."):
                self.liquidations.add(parse_bybit_all_liquidation(msg))
            if msg.get("op") == "subscribe" and msg.get("success"):
                self._bybit_liq_connected = True
            if self._bybit_liq_connected:
                self.liquidations.heartbeat(now_ms)

        return _handler

    async def _periodic(self) -> None:
        last_flush = time.monotonic()
        contract_values: dict[str, float] = {}
        while not self._stop.is_set():
            now_ms = int(time.time() * 1000)
            try:
                self.boundaries.update(now_ms)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Boundary imbalance update failed: %s", exc)
            if self._okx_poll and not self._bybit_liq_connected:
                try:
                    if not contract_values:
                        contract_values = await asyncio.to_thread(fetch_okx_contract_values, [f"{c}-USDT-SWAP" for c in self.coins])
                    for c in self.coins:
                        events = await asyncio.to_thread(fetch_okx_liquidations, c, contract_values)
                        self.liquidations.add(events)
                        self.liquidations.heartbeat(now_ms, [c])
                    self._status["okx_liquidation_poll"] = "ok"
                except Exception as exc:
                    self._status["okx_liquidation_poll"] = f"failed ({type(exc).__name__})"
                    self.liquidations.break_coverage()
            if self._trades_dir is not None and time.monotonic() - last_flush >= self._flush_interval_s:
                try:
                    await asyncio.to_thread(self.tape.flush_parquet, self._trades_dir)
                except Exception as exc:
                    logger.warning("Trade tape flush failed: %s", exc)
                last_flush = time.monotonic()
            await asyncio.sleep(5.0)


def replay_recorded_messages(hub: StreamHub, recording: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Feed a recording (``{stream_name: [{"recv_ms", "raw"}]}``) through the hub's handlers in time order."""
    bybit = hub._make_bybit_handler()
    handlers: dict[str, Callable[[Mapping[str, Any], int], None]] = {
        "binance_bookticker": hub._on_binance,
        "bybit_orderbook1": bybit,
        "bybit_all_liquidation": bybit,
        "hyperliquid_trades": hub._on_hl,
    }
    events: list[tuple[int, str, Mapping[str, Any]]] = []
    for name, rows in recording.items():
        for row in rows:
            events.append((int(row["recv_ms"]), name, row["raw"]))
    events.sort(key=lambda x: x[0])
    for recv_ms, name, raw in events:
        h = handlers.get(name)
        if h is not None and isinstance(raw, dict):
            h(raw, recv_ms)


__all__ = [
    "BOUNDARY_WINDOWS_S",
    "BarFlow",
    "BoundaryImbalance",
    "BoundaryImbalanceTracker",
    "BybitTopOfBook",
    "Coverage",
    "LeadMidBuffer",
    "LiquidationBook",
    "LiquidationEvent",
    "StreamAvailability",
    "StreamHub",
    "Trade",
    "TradeTape",
    "boundary_imbalance_from_tape",
    "compute_boundary_imbalance",
    "fetch_okx_contract_values",
    "fetch_okx_liquidations",
    "parse_binance_book_ticker",
    "parse_bybit_all_liquidation",
    "parse_hyperliquid_trades",
    "parse_okx_liquidation_orders",
    "quarter_hour_boundaries",
    "replay_recorded_messages",
]
