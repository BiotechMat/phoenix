"""Multi-asset historical backfill from real venues (REINFORCEMENT_PLAN task P1.2).

Sources, all real:

- Candles (15m and 1h): Bybit V5 ``kline`` (primary) or OKX ``history-candles`` (fallback; the
  only one of the two reachable from some sandboxes). Both keep years of history.
- Mark and index prices: Bybit ``mark-price-kline`` / ``index-price-kline`` or OKX
  ``history-mark-price-candles`` / ``history-index-candles`` so the basis is real, not a proxy.
- Funding: Hyperliquid ``fundingHistory`` (hourly settlements, multi-year). This is the venue the
  carry sleeve trades, so its funding is the one that matters for labels and carry PnL.
- Open interest: Bybit ``open-interest`` history (primary, multi-year) or OKX
  ``open-interest-history`` (recent weeks only).
- Liquidations and taker buy volume: no history feed yet (P1.1 / P1.3). The columns are ``0.0`` and
  ``null`` respectively and ``fields_unavailable`` names them.

Nothing is synthesised. A bar whose open interest is unknown carries ``null`` and is excluded from
model training by the live loop's Parquet loader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Final, Literal
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import polars as pl
import pyarrow as pa

logger = logging.getLogger("phoenix.data.backfill")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BTC_PARQUET_FILE = DATA_DIR / "btcusdt_perp_bars.parquet"
ETH_PARQUET_FILE = DATA_DIR / "ethusdt_perp_bars.parquet"
SOL_PARQUET_FILE = DATA_DIR / "solusdt_perp_bars.parquet"

BASE_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "horizon_label",
    "ts_open_ns",
    "ts_close_ns",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "buy_dollar_volume",
    "sell_dollar_volume",
    "open_interest",
    "funding_rate_8h",
    "mark_price",
    "spot_price",
    "liquidation_volume_usd",
    "tick_count",
)

ARROW_BASE_SCHEMA = pa.schema([
    ("instrument_id", pa.string()),
    ("horizon_label", pa.string()),
    ("ts_open_ns", pa.int64()),
    ("ts_close_ns", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("dollar_volume", pa.float64()),
    ("buy_dollar_volume", pa.float64()),
    ("sell_dollar_volume", pa.float64()),
    ("open_interest", pa.float64()),
    ("funding_rate_8h", pa.float64()),
    ("mark_price", pa.float64()),
    ("spot_price", pa.float64()),
    ("liquidation_volume_usd", pa.float64()),
    ("tick_count", pa.int64()),
])

EXTRA_COLUMNS: tuple[str, ...] = (
    "funding_rate_1h",
    "taker_buy_volume",
    "bar_source",
    "oi_source",
    "fields_unavailable",
)

Horizon = Literal["15m", "1h"]
Venue = Literal["bybit", "okx"]

HORIZON_MS: Final[dict[str, int]] = {"15m": 900_000, "1h": 3_600_000}
BYBIT_INTERVAL: Final[dict[str, str]] = {"15m": "15", "1h": "60"}
BYBIT_OI_INTERVAL: Final[dict[str, str]] = {"15m": "15min", "1h": "1h"}
OKX_BAR: Final[dict[str, str]] = {"15m": "15m", "1h": "1H"}
OKX_OI_PERIOD: Final[dict[str, str]] = {"15m": "15m", "1h": "1H"}

BYBIT_BASE_URL: Final[str] = "https://api.bybit.com/v5/market"
OKX_BASE_URL: Final[str] = "https://www.okx.com/api/v5"
HYPERLIQUID_INFO_URL: Final[str] = "https://api.hyperliquid.xyz/info"

BYBIT_PAGE_LIMIT: Final[int] = 1000
BYBIT_OI_PAGE_LIMIT: Final[int] = 200
OKX_PAGE_LIMIT: Final[int] = 100
HYPERLIQUID_FUNDING_PAGE: Final[int] = 500
MAX_PAGES: Final[int] = 5000
DEFAULT_TIMEOUT_SEC: Final[float] = 10.0
DEFAULT_PAUSE_SEC: Final[float] = 0.12
MAX_ALLOWED_GAP_BARS: Final[int] = 2


@dataclass(frozen=True)
class Candle:
    """One closed candle in base-asset units."""

    t_ms: int
    open: float
    high: float
    low: float
    close: float
    volume_base: float
    quote_volume: float
    n_trades: int | None = None


@dataclass(frozen=True)
class CoverageReport:
    """Row count, date coverage and gap statistics for one asset and horizon."""

    coin: str
    horizon: str
    rows: int
    first_ts_ms: int | None
    last_ts_ms: int | None
    expected_rows: int
    missing_rows: int
    max_gap_bars: int
    gaps: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    oi_rows: int = 0
    funding_rows: int = 0
    mark_rows: int = 0
    candle_source: str = ""
    oi_source: str = ""
    path: str = ""

    @property
    def passes_gap_rule(self) -> bool:
        """True when no gap exceeds ``MAX_ALLOWED_GAP_BARS``."""
        return self.max_gap_bars <= MAX_ALLOWED_GAP_BARS

    def as_dict(self) -> dict[str, Any]:
        return {
            "coin": self.coin,
            "horizon": self.horizon,
            "rows": self.rows,
            "first_ts_ms": self.first_ts_ms,
            "last_ts_ms": self.last_ts_ms,
            "expected_rows": self.expected_rows,
            "missing_rows": self.missing_rows,
            "max_gap_bars": self.max_gap_bars,
            "gaps": [list(g) for g in self.gaps],
            "oi_rows": self.oi_rows,
            "funding_rows": self.funding_rows,
            "mark_rows": self.mark_rows,
            "candle_source": self.candle_source,
            "oi_source": self.oi_source,
            "passes_gap_rule": self.passes_gap_rule,
            "path": self.path,
        }


# ------------------------------------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------------------------------------


def _http_get_json(url: str, params: Mapping[str, Any], timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Any:
    query = urllib.parse.urlencode({k: str(v) for k, v in params.items() if v is not None})
    req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "ProjectPhoenix/4.1"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: Mapping[str, Any], timeout_sec: float = DEFAULT_TIMEOUT_SEC, max_retries: int = 6) -> Any:
    """POST with exponential backoff on HTTP 429 (rate limited) and 5xx."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "ProjectPhoenix/4.1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if attempt >= max_retries or not (exc.code == 429 or exc.code >= 500):
                raise
            logger.info("HTTP %s from %s; retrying in %.0fs", exc.code, url, delay)
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
    raise RuntimeError("unreachable")


# ------------------------------------------------------------------------------------------------
# Parsers (pure, fixture-testable)
# ------------------------------------------------------------------------------------------------


def parse_bybit_klines(rows: Sequence[Sequence[Any]]) -> list[Candle]:
    """Bybit V5 kline rows ``[startTime, open, high, low, close, volume, turnover]`` -> candles, oldest first."""
    out: list[Candle] = []
    for r in rows:
        if len(r) < 7:
            continue
        out.append(
            Candle(
                t_ms=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume_base=float(r[5]),
                quote_volume=float(r[6]),
            )
        )
    out.sort(key=lambda c: c.t_ms)
    return out


def parse_bybit_price_klines(rows: Sequence[Sequence[Any]]) -> list[tuple[int, float]]:
    """Bybit mark/index price kline rows ``[startTime, open, high, low, close]`` -> ``(t_ms, close)``."""
    out = [(int(r[0]), float(r[4])) for r in rows if len(r) >= 5]
    out.sort(key=lambda x: x[0])
    return out


def parse_okx_candles(rows: Sequence[Sequence[Any]], *, require_confirmed: bool = True) -> list[Candle]:
    """OKX candle rows ``[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]`` -> candles, oldest first.

    ``volCcy`` is the base-asset volume (contract multipliers already applied by the venue).
    """
    out: list[Candle] = []
    for r in rows:
        if len(r) < 8:
            continue
        if require_confirmed and len(r) >= 9 and str(r[8]) != "1":
            continue
        out.append(
            Candle(
                t_ms=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume_base=float(r[6]),
                quote_volume=float(r[7]),
            )
        )
    out.sort(key=lambda c: c.t_ms)
    return out


def parse_okx_price_candles(rows: Sequence[Sequence[Any]]) -> list[tuple[int, float]]:
    """OKX mark/index price candle rows ``[ts, o, h, l, c, confirm]`` -> ``(t_ms, close)``."""
    out: list[tuple[int, float]] = []
    for r in rows:
        if len(r) < 5:
            continue
        if len(r) >= 6 and str(r[5]) != "1":
            continue
        out.append((int(r[0]), float(r[4])))
    out.sort(key=lambda x: x[0])
    return out


def parse_bybit_open_interest(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, float]]:
    """Bybit open-interest rows ``{"openInterest", "timestamp"}`` -> ``(t_ms, oi_base)`` oldest first."""
    out = [(int(r["timestamp"]), float(r["openInterest"])) for r in rows if "timestamp" in r and "openInterest" in r]
    out.sort(key=lambda x: x[0])
    return out


def parse_okx_open_interest(rows: Sequence[Sequence[Any]]) -> list[tuple[int, float]]:
    """OKX open-interest-history rows ``[ts, oi_contracts, oi_ccy, oi_usd]`` -> ``(t_ms, oi_base)``."""
    out = [(int(r[0]), float(r[2])) for r in rows if len(r) >= 3]
    out.sort(key=lambda x: x[0])
    return out


def parse_hyperliquid_funding(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, float]]:
    """Hyperliquid fundingHistory rows ``{"time", "fundingRate"}`` -> ``(t_ms, rate_per_settlement)`` oldest first."""
    out = [(int(r["time"]), float(r["fundingRate"])) for r in rows if "time" in r and "fundingRate" in r]
    out.sort(key=lambda x: x[0])
    return out


def normalize_funding_to_hourly(series: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    """Convert each settlement's rate to a per-hour rate using the interval since the previous settlement.

    Hyperliquid's early history settles every 8 hours (rates such as 0.0001 per 8 hours) and its later
    history every hour (0.0000125 per hour for the same base rate), so the raw values are not comparable
    without this step. The first settlement uses the interval to the next one.
    """
    pts = sorted(series)
    out: list[tuple[int, float]] = []
    for i, (t, rate) in enumerate(pts):
        if i > 0:
            gap_ms = t - pts[i - 1][0]
        elif len(pts) > 1:
            gap_ms = pts[1][0] - t
        else:
            gap_ms = 3_600_000
        hours = max(1, int(round(gap_ms / 3_600_000)))
        out.append((t, rate / float(hours)))
    return out


# ------------------------------------------------------------------------------------------------
# Fetchers (paged)
# ------------------------------------------------------------------------------------------------


def fetch_bybit_klines(
    symbol: str,
    horizon: Horizon,
    start_ms: int,
    end_ms: int,
    *,
    kind: Literal["kline", "mark-price-kline", "index-price-kline"] = "kline",
    base_url: str = BYBIT_BASE_URL,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    pause_sec: float = DEFAULT_PAUSE_SEC,
) -> list[Sequence[Any]]:
    """Page Bybit klines backwards from ``end_ms`` to ``start_ms``; returns raw rows oldest first."""
    rows: list[Sequence[Any]] = []
    cursor_end = int(end_ms)
    for _ in range(MAX_PAGES):
        data = _http_get_json(
            f"{base_url}/{kind}",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": BYBIT_INTERVAL[horizon],
                "start": int(start_ms),
                "end": cursor_end,
                "limit": BYBIT_PAGE_LIMIT,
            },
            timeout_sec=timeout_sec,
        )
        page = list(data.get("result", {}).get("list", []))
        if not page:
            break
        rows.extend(page)
        oldest = min(int(r[0]) for r in page)
        if oldest <= start_ms or len(page) < BYBIT_PAGE_LIMIT:
            break
        cursor_end = oldest - 1
        time.sleep(pause_sec)
    rows.sort(key=lambda r: int(r[0]))
    return [r for r in rows if start_ms <= int(r[0]) <= end_ms]


def fetch_okx_candles(
    inst_id: str,
    horizon: Horizon,
    start_ms: int,
    end_ms: int,
    *,
    kind: Literal["history-candles", "history-mark-price-candles", "history-index-candles"] = "history-candles",
    base_url: str = OKX_BASE_URL,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    pause_sec: float = DEFAULT_PAUSE_SEC,
) -> list[Sequence[Any]]:
    """Page OKX candles backwards (``after`` = older than) from ``end_ms`` to ``start_ms``; raw rows oldest first."""
    rows: list[Sequence[Any]] = []
    cursor_after = int(end_ms) + 1
    for _ in range(MAX_PAGES):
        data = _http_get_json(
            f"{base_url}/market/{kind}",
            {"instId": inst_id, "bar": OKX_BAR[horizon], "after": cursor_after, "limit": OKX_PAGE_LIMIT},
            timeout_sec=timeout_sec,
        )
        if str(data.get("code", "")) != "0":
            raise RuntimeError(f"OKX {kind} error for {inst_id}: {data.get('msg', data)}")
        page = list(data.get("data", []))
        if not page:
            break
        rows.extend(page)
        oldest = min(int(r[0]) for r in page)
        if oldest <= start_ms:
            break
        cursor_after = oldest
        time.sleep(pause_sec)
    rows.sort(key=lambda r: int(r[0]))
    return [r for r in rows if start_ms <= int(r[0]) <= end_ms]


def fetch_bybit_open_interest(
    symbol: str,
    horizon: Horizon,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = BYBIT_BASE_URL,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    pause_sec: float = DEFAULT_PAUSE_SEC,
) -> list[tuple[int, float]]:
    """Page Bybit open-interest history with its cursor; ``(t_ms, oi)`` oldest first."""
    rows: list[Mapping[str, Any]] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": BYBIT_OI_INTERVAL[horizon],
            "startTime": int(start_ms),
            "endTime": int(end_ms),
            "limit": BYBIT_OI_PAGE_LIMIT,
        }
        if cursor:
            params["cursor"] = cursor
        data = _http_get_json(f"{base_url}/open-interest", params, timeout_sec=timeout_sec)
        result = data.get("result", {})
        page = list(result.get("list", []))
        if not page:
            break
        rows.extend(page)
        cursor = result.get("nextPageCursor") or None
        if not cursor:
            break
        time.sleep(pause_sec)
    return [(t, oi) for t, oi in parse_bybit_open_interest(rows) if start_ms <= t <= end_ms]


def fetch_okx_open_interest(
    inst_id: str,
    horizon: Horizon,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = OKX_BASE_URL,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    pause_sec: float = DEFAULT_PAUSE_SEC,
) -> list[tuple[int, float]]:
    """Page OKX open-interest-history backwards; only recent weeks exist on the venue."""
    rows: list[Sequence[Any]] = []
    cursor_end = int(end_ms)
    for _ in range(MAX_PAGES):
        data = _http_get_json(
            f"{base_url}/rubik/stat/contracts/open-interest-history",
            {"instId": inst_id, "period": OKX_OI_PERIOD[horizon], "end": cursor_end, "limit": OKX_PAGE_LIMIT},
            timeout_sec=timeout_sec,
        )
        if str(data.get("code", "")) != "0":
            raise RuntimeError(f"OKX open-interest-history error for {inst_id}: {data.get('msg', data)}")
        page = list(data.get("data", []))
        if not page:
            break
        rows.extend(page)
        oldest = min(int(r[0]) for r in page)
        if oldest <= start_ms:
            break
        cursor_end = oldest - 1
        time.sleep(pause_sec)
    return [(t, oi) for t, oi in parse_okx_open_interest(rows) if start_ms <= t <= end_ms]


def fetch_hyperliquid_funding(
    coin: str,
    start_ms: int,
    end_ms: int,
    *,
    url: str = HYPERLIQUID_INFO_URL,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    pause_sec: float = DEFAULT_PAUSE_SEC,
) -> list[tuple[int, float]]:
    """Page Hyperliquid fundingHistory forwards in 500-row pages; ``(t_ms, rate_per_hour)`` oldest first."""
    rows: list[Mapping[str, Any]] = []
    cursor_start = int(start_ms)
    for _ in range(MAX_PAGES):
        data = _http_post_json(
            url,
            {"type": "fundingHistory", "coin": coin, "startTime": cursor_start, "endTime": int(end_ms)},
            timeout_sec=timeout_sec,
        )
        page = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        if not page:
            break
        rows.extend(page)
        newest = max(int(r["time"]) for r in page)
        if newest >= end_ms or len(page) < HYPERLIQUID_FUNDING_PAGE:
            break
        cursor_start = newest + 1
        time.sleep(max(pause_sec, 0.5))  # Hyperliquid rate-limits info requests by weight
    seen: dict[int, float] = {}
    for t, rate in parse_hyperliquid_funding(rows):
        seen[t] = rate
    hourly = normalize_funding_to_hourly(sorted(seen.items()))
    return [(t, r) for t, r in hourly if start_ms <= t <= end_ms]


# ------------------------------------------------------------------------------------------------
# Assembly, merge, coverage
# ------------------------------------------------------------------------------------------------


def _causal_values(
    series: Sequence[tuple[int, float]],
    query_ts_ms: np.ndarray,
    *,
    max_staleness_ms: int | None,
) -> np.ndarray:
    """Value at or before each query timestamp (NaN when none, or when staler than ``max_staleness_ms``)."""
    out = np.full(query_ts_ms.shape[0], np.nan, dtype=np.float64)
    if not series:
        return out
    ts = np.array([t for t, _ in series], dtype=np.int64)
    vals = np.array([v for _, v in series], dtype=np.float64)
    idx = np.searchsorted(ts, query_ts_ms, side="right") - 1
    ok = idx >= 0
    if max_staleness_ms is not None:
        ok &= (query_ts_ms - ts[np.maximum(idx, 0)]) <= max_staleness_ms
    out[ok] = vals[idx[ok]]
    return out


def perp_instrument_id(coin: str) -> str:
    return f"{coin.upper()}USDT-PERP"


def assemble_bars(
    coin: str,
    horizon: Horizon,
    candles: Sequence[Candle],
    *,
    funding: Sequence[tuple[int, float]] = (),
    open_interest: Sequence[tuple[int, float]] = (),
    mark_closes: Sequence[tuple[int, float]] = (),
    index_closes: Sequence[tuple[int, float]] = (),
    candle_source: str = "",
    oi_source: str = "",
) -> pl.DataFrame:
    """Build the 18-column bar table plus provenance columns from real series.

    - ``funding_rate_1h`` is the last Hyperliquid settlement at or before the bar open (causal).
    - ``open_interest`` is the last observation at or before the bar close, ``null`` when none
      exists within two bar lengths.
    - ``mark_price`` / ``spot_price`` come from mark and index candles for the same bar when
      available, otherwise the close (listed in ``fields_unavailable``).
    - ``buy_dollar_volume`` / ``sell_dollar_volume`` are an even split; ``taker_buy_volume`` is null.
    - ``liquidation_volume_usd`` is ``0.0`` and listed as unavailable.
    """
    step_ms = HORIZON_MS[horizon]
    cands = sorted(candles, key=lambda c: c.t_ms)
    n = len(cands)
    t_open = np.array([c.t_ms for c in cands], dtype=np.int64)
    t_close = t_open + step_ms
    closes = np.array([c.close for c in cands], dtype=np.float64)

    funding_1h = _causal_values(funding, t_open, max_staleness_ms=None)
    funding_1h = np.where(np.isnan(funding_1h), 0.0, funding_1h)
    funding_known = int(np.sum(~np.isnan(_causal_values(funding, t_open, max_staleness_ms=None)))) if funding else 0

    oi = _causal_values(open_interest, t_close, max_staleness_ms=2 * step_ms)
    mark = _causal_values(mark_closes, t_open, max_staleness_ms=0) if mark_closes else np.full(n, np.nan)
    index = _causal_values(index_closes, t_open, max_staleness_ms=0) if index_closes else np.full(n, np.nan)

    missing: list[str] = ["taker_flow", "liquidations"]
    if not mark_closes:
        missing.append("mark_price")
    if not index_closes:
        missing.append("spot_price")
    if not funding:
        missing.append("funding")
    unavailable_tag = ",".join(missing)

    mark_filled = np.where(np.isnan(mark), closes, mark)
    index_filled = np.where(np.isnan(index), closes, index)
    volumes = np.array([c.volume_base for c in cands], dtype=np.float64)
    dvol = np.array([c.quote_volume if c.quote_volume > 0.0 else c.volume_base * c.close for c in cands], dtype=np.float64)

    df = pl.DataFrame(
        {
            "instrument_id": [perp_instrument_id(coin)] * n,
            "horizon_label": [horizon] * n,
            "ts_open_ns": (t_open * 1_000_000).tolist(),
            "ts_close_ns": (t_close * 1_000_000).tolist(),
            "open": [c.open for c in cands],
            "high": [max(c.high, c.open, c.close) for c in cands],
            "low": [min(c.low, c.open, c.close) for c in cands],
            "close": closes.tolist(),
            "volume": volumes.tolist(),
            "dollar_volume": dvol.tolist(),
            "buy_dollar_volume": (dvol * 0.5).tolist(),
            "sell_dollar_volume": (dvol * 0.5).tolist(),
            "open_interest": [None if np.isnan(v) else float(v) for v in oi],
            "funding_rate_8h": (funding_1h * 8.0).tolist(),
            "mark_price": mark_filled.tolist(),
            "spot_price": index_filled.tolist(),
            "liquidation_volume_usd": [0.0] * n,
            "tick_count": [int(c.n_trades) if c.n_trades is not None and c.n_trades > 0 else 1 for c in cands],
            "funding_rate_1h": funding_1h.tolist(),
            "taker_buy_volume": [None] * n,
            "bar_source": [f"BACKFILL-{candle_source.upper()}" if candle_source else "BACKFILL"] * n,
            "oi_source": [oi_source.upper() if oi_source else ""] * n,
            "fields_unavailable": [unavailable_tag] * n,
        },
        schema_overrides={
            "ts_open_ns": pl.Int64,
            "ts_close_ns": pl.Int64,
            "open_interest": pl.Float64,
            "taker_buy_volume": pl.Float64,
            "tick_count": pl.Int64,
        },
    )
    _ = funding_known
    return df


def merge_into_parquet(df: pl.DataFrame, path: Path) -> int:
    """Union ``df`` with the existing file on ``ts_close_ns``, preferring rows that carry open interest.

    Returns the resulting row count. Existing live rows (which have real open interest) are never
    replaced by a backfilled row without it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, df], how="diagonal_relaxed")
    else:
        combined = df
    if "open_interest" not in combined.columns:
        combined = combined.with_columns(pl.lit(None, dtype=pl.Float64).alias("open_interest"))
    combined = (
        combined.with_row_index("_order")
        .with_columns(pl.col("open_interest").is_not_null().cast(pl.Int8).alias("_has_oi"))
        .sort(["ts_close_ns", "_has_oi", "_order"])
        .unique(subset=["ts_close_ns"], keep="last", maintain_order=True)
        .drop(["_order", "_has_oi"])
        .sort("ts_close_ns")
    )
    combined.write_parquet(path)
    return int(combined.height)


def coverage_report(df: pl.DataFrame, coin: str, horizon: Horizon, *, path: Path | None = None) -> CoverageReport:
    """Row counts, date coverage and gaps (in bars) of a bar table."""
    step_ns = HORIZON_MS[horizon] * 1_000_000
    if df.height == 0:
        return CoverageReport(coin=coin, horizon=horizon, rows=0, first_ts_ms=None, last_ts_ms=None, expected_rows=0, missing_rows=0, max_gap_bars=0, path=str(path) if path else "")
    ts = np.sort(df.get_column("ts_open_ns").to_numpy().astype(np.int64))
    diffs = np.diff(ts)
    gap_bars = np.round(diffs / step_ns).astype(np.int64) - 1
    gaps = tuple((int(ts[i] // 1_000_000), int(g)) for i, g in enumerate(gap_bars) if g > 0)
    expected = int((ts[-1] - ts[0]) // step_ns) + 1
    oi_rows = int(df.get_column("open_interest").is_not_null().sum()) if "open_interest" in df.columns else 0
    funding_rows = int((df.get_column("funding_rate_1h") != 0.0).sum()) if "funding_rate_1h" in df.columns else 0
    mark_rows = int((df.get_column("mark_price") != df.get_column("close")).sum()) if "mark_price" in df.columns else 0
    return CoverageReport(
        coin=coin,
        horizon=horizon,
        rows=int(df.height),
        first_ts_ms=int(ts[0] // 1_000_000),
        last_ts_ms=int(ts[-1] // 1_000_000),
        expected_rows=expected,
        missing_rows=max(0, expected - int(len(ts))),
        max_gap_bars=int(gap_bars.max()) if gap_bars.size else 0,
        gaps=gaps,
        oi_rows=oi_rows,
        funding_rows=funding_rows,
        mark_rows=mark_rows,
        candle_source=str(df.get_column("bar_source")[0]) if "bar_source" in df.columns else "",
        oi_source=str(df.get_column("oi_source")[0]) if "oi_source" in df.columns else "",
        path=str(path) if path else "",
    )


def parquet_path_for(coin: str, horizon: Horizon, data_dir: Path = DATA_DIR) -> Path:
    """``data/{coin}usdt_perp_bars.parquet`` for 15m (the live training store) and ``..._1h.parquet`` for 1h."""
    stem = f"{coin.lower()}usdt_perp_bars"
    return data_dir / (f"{stem}.parquet" if horizon == "15m" else f"{stem}_{horizon}.parquet")


# ------------------------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------------------------


def _bybit_symbol(coin: str) -> str:
    return f"{coin.upper()}USDT"


def _okx_swap(coin: str) -> str:
    return f"{coin.upper()}-USDT-SWAP"


def _okx_index(coin: str) -> str:
    return f"{coin.upper()}-USDT"


def fetch_candle_bundle(
    coin: str,
    horizon: Horizon,
    start_ms: int,
    end_ms: int,
    *,
    venues: Sequence[Venue] = ("bybit", "okx"),
    pause_sec: float = DEFAULT_PAUSE_SEC,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    include_mark_index: bool = True,
) -> tuple[list[Candle], list[tuple[int, float]], list[tuple[int, float]], str]:
    """Candles plus mark and index closes from the first venue that answers; returns the venue used.

    With ``include_mark_index=False`` only trade candles are fetched (mark and spot then fall back to the
    close and are listed in ``fields_unavailable``).
    """
    last_error: Exception | None = None
    for venue in venues:
        try:
            if venue == "bybit":
                sym = _bybit_symbol(coin)
                candles = parse_bybit_klines(fetch_bybit_klines(sym, horizon, start_ms, end_ms, pause_sec=pause_sec, timeout_sec=timeout_sec))
                if not candles:
                    continue
                if not include_mark_index:
                    return candles, [], [], "bybit"
                marks = parse_bybit_price_klines(fetch_bybit_klines(sym, horizon, start_ms, end_ms, kind="mark-price-kline", pause_sec=pause_sec, timeout_sec=timeout_sec))
                index = parse_bybit_price_klines(fetch_bybit_klines(sym, horizon, start_ms, end_ms, kind="index-price-kline", pause_sec=pause_sec, timeout_sec=timeout_sec))
                return candles, marks, index, "bybit"
            if venue == "okx":
                candles = parse_okx_candles(fetch_okx_candles(_okx_swap(coin), horizon, start_ms, end_ms, pause_sec=pause_sec, timeout_sec=timeout_sec))
                if not candles:
                    continue
                if not include_mark_index:
                    return candles, [], [], "okx"
                marks = parse_okx_price_candles(fetch_okx_candles(_okx_swap(coin), horizon, start_ms, end_ms, kind="history-mark-price-candles", pause_sec=pause_sec, timeout_sec=timeout_sec))
                index = parse_okx_price_candles(fetch_okx_candles(_okx_index(coin), horizon, start_ms, end_ms, kind="history-index-candles", pause_sec=pause_sec, timeout_sec=timeout_sec))
                return candles, marks, index, "okx"
        except Exception as exc:
            last_error = exc
            logger.warning("Candle fetch from %s failed for %s %s: %s", venue, coin, horizon, exc)
    if last_error is not None:
        logger.warning("No candle venue answered for %s %s (last error: %s)", coin, horizon, last_error)
    return [], [], [], ""


def fetch_open_interest_bundle(
    coin: str,
    horizon: Horizon,
    start_ms: int,
    end_ms: int,
    *,
    venues: Sequence[Venue] = ("bybit", "okx"),
    pause_sec: float = DEFAULT_PAUSE_SEC,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> tuple[list[tuple[int, float]], str]:
    """Open-interest history from the first venue that answers; returns the venue used."""
    for venue in venues:
        try:
            if venue == "bybit":
                series = fetch_bybit_open_interest(_bybit_symbol(coin), horizon, start_ms, end_ms, pause_sec=pause_sec, timeout_sec=timeout_sec)
            else:
                series = fetch_okx_open_interest(_okx_swap(coin), horizon, start_ms, end_ms, pause_sec=pause_sec, timeout_sec=timeout_sec)
            if series:
                return series, venue
        except Exception as exc:
            logger.warning("Open-interest fetch from %s failed for %s %s: %s", venue, coin, horizon, exc)
    return [], ""


def backfill_asset(
    coin: str,
    horizon: Horizon,
    start_ms: int,
    end_ms: int,
    *,
    data_dir: Path = DATA_DIR,
    venues: Sequence[Venue] = ("bybit", "okx"),
    pause_sec: float = DEFAULT_PAUSE_SEC,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    write: bool = True,
    include_mark_index: bool = True,
    include_open_interest: bool = True,
) -> tuple[pl.DataFrame, CoverageReport]:
    """Fetch, assemble, merge and report one asset at one horizon."""
    candles, marks, index, candle_venue = fetch_candle_bundle(
        coin, horizon, start_ms, end_ms, venues=venues, pause_sec=pause_sec, timeout_sec=timeout_sec, include_mark_index=include_mark_index
    )
    if not candles:
        empty = assemble_bars(coin, horizon, [])
        return empty, coverage_report(empty, coin, horizon)
    try:
        funding = fetch_hyperliquid_funding(coin.upper(), start_ms - 3_600_000, end_ms, pause_sec=pause_sec, timeout_sec=timeout_sec)
    except Exception as exc:
        logger.warning("Hyperliquid funding history failed for %s: %s", coin, exc)
        funding = []
    oi_series: list[tuple[int, float]] = []
    oi_venue = ""
    if include_open_interest:
        oi_series, oi_venue = fetch_open_interest_bundle(coin, horizon, start_ms, end_ms, venues=venues, pause_sec=pause_sec, timeout_sec=timeout_sec)

    df = assemble_bars(
        coin,
        horizon,
        candles,
        funding=funding,
        open_interest=oi_series,
        mark_closes=marks,
        index_closes=index,
        candle_source=candle_venue,
        oi_source=oi_venue,
    )
    path = parquet_path_for(coin, horizon, data_dir)
    if write:
        merge_into_parquet(df, path)
    report = coverage_report(df, coin, horizon, path=path)
    logger.info(
        "Backfilled %s %s: %d rows from %s (%s to %s), OI rows %d from %s, funding rows %d, max gap %d bars",
        coin, horizon, report.rows, candle_venue, report.first_ts_ms, report.last_ts_ms, report.oi_rows, oi_venue or "none", report.funding_rows, report.max_gap_bars,
    )
    return df, report


def refresh_funding(coin: str, horizon: Horizon, *, data_dir: Path = DATA_DIR, pause_sec: float = DEFAULT_PAUSE_SEC) -> int:
    """Re-fetch Hyperliquid funding for an existing Parquet and rewrite its funding columns causally.

    Returns the number of bars that now carry a funding settlement.
    """
    path = parquet_path_for(coin, horizon, data_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path).sort("ts_open_ns")
    t_open = (df.get_column("ts_open_ns").to_numpy() // 1_000_000).astype(np.int64)
    funding = fetch_hyperliquid_funding(coin.upper(), int(t_open[0]) - 8 * 3_600_000, int(t_open[-1]), pause_sec=pause_sec)
    f1h = _causal_values(funding, t_open, max_staleness_ms=None)
    known = ~np.isnan(f1h)
    f1h = np.where(known, f1h, 0.0)
    unavailable = df.get_column("fields_unavailable").to_list() if "fields_unavailable" in df.columns else [""] * df.height
    cleaned = [",".join(x for x in str(u).split(",") if x and x != "funding") + ("" if k else ",funding") for u, k in zip(unavailable, known)]
    df = df.with_columns(
        pl.Series("funding_rate_1h", f1h.tolist()),
        pl.Series("funding_rate_8h", (f1h * 8.0).tolist()),
        pl.Series("fields_unavailable", [c.strip(",") for c in cleaned]),
    )
    df.write_parquet(path)
    return int(known.sum())


def run_backfill(
    coins: Sequence[str] = ("BTC", "ETH", "SOL"),
    horizons: Sequence[Horizon] = ("15m", "1h"),
    *,
    start_ms: int,
    end_ms: int,
    data_dir: Path = DATA_DIR,
    venues: Sequence[Venue] = ("bybit", "okx"),
    pause_sec: float = DEFAULT_PAUSE_SEC,
    include_mark_index: bool = True,
    include_open_interest: bool = True,
) -> dict[str, CoverageReport]:
    """Backfill every coin and horizon; returns coverage reports keyed ``COIN:horizon``."""
    reports: dict[str, CoverageReport] = {}
    for coin in coins:
        for horizon in horizons:
            _, report = backfill_asset(
                coin, horizon, start_ms, end_ms, data_dir=data_dir, venues=venues, pause_sec=pause_sec,
                include_mark_index=include_mark_index, include_open_interest=include_open_interest,
            )
            reports[f"{coin.upper()}:{horizon}"] = report
    return reports


__all__ = [
    "ARROW_BASE_SCHEMA",
    "BASE_COLUMNS",
    "BTC_PARQUET_FILE",
    "Candle",
    "CoverageReport",
    "DATA_DIR",
    "ETH_PARQUET_FILE",
    "EXTRA_COLUMNS",
    "HORIZON_MS",
    "MAX_ALLOWED_GAP_BARS",
    "SOL_PARQUET_FILE",
    "assemble_bars",
    "backfill_asset",
    "coverage_report",
    "fetch_bybit_klines",
    "fetch_bybit_open_interest",
    "fetch_candle_bundle",
    "fetch_hyperliquid_funding",
    "fetch_okx_candles",
    "fetch_okx_open_interest",
    "fetch_open_interest_bundle",
    "merge_into_parquet",
    "parquet_path_for",
    "parse_bybit_klines",
    "parse_bybit_open_interest",
    "parse_bybit_price_klines",
    "parse_hyperliquid_funding",
    "parse_okx_candles",
    "parse_okx_open_interest",
    "parse_okx_price_candles",
    "normalize_funding_to_hourly",
    "perp_instrument_id",
    "refresh_funding",
    "run_backfill",
]
