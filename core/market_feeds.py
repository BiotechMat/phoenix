"""Auxiliary live feeds with TTL caching (REINFORCEMENT_PLAN tasks P1.4 and P0.8).

Covers, with pure parsers that are tested on recorded fixtures:

- Hyperliquid ``predictedFundings`` (cross-venue predicted funding per asset), ``perpsAtOpenInterestCap``
  and ``perpDexLimits``.
- Deribit ``get_volatility_index_data`` (DVOL), ``get_instruments`` and ``ticker`` (mark IV, delta, greeks).
- Stablecoin marks for USDT, USDC and USDe from several venues, reduced to a per-symbol median and a
  peg deviation in basis points; a symbol counts only when at least two venues quote it.

``LiveMacroInputsFeed`` turns DVOL and the stablecoin marks into the inputs the macro sentinel
needs. Anything that cannot be fetched is ``None`` and reported in the availability flags; no
constant stands in for a missing feed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import json
import logging
import math
import statistics
import time
from typing import Any, Final, Literal, Protocol
import urllib.parse
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("phoenix.market_feeds")

HYPERLIQUID_INFO_URL: Final[str] = "https://api.hyperliquid.xyz/info"
DERIBIT_BASE_URL: Final[str] = "https://www.deribit.com/api/v2/public"
DEFAULT_TIMEOUT_SEC: Final[float] = 4.0
MINUTE_NS: Final[int] = 60 * 1_000_000_000
DEFAULT_TTL_NS: Final[int] = 5 * MINUTE_NS
DVOL_LOOKBACK_HOURS: Final[int] = 30 * 24


# ------------------------------------------------------------------------------------------------
# HTTP + cache
# ------------------------------------------------------------------------------------------------


def _get_json(url: str, params: Mapping[str, Any] | None = None, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Any:
    full = f"{url}?{urllib.parse.urlencode({k: str(v) for k, v in (params or {}).items()})}" if params else url
    req = urllib.request.Request(full, headers={"User-Agent": "ProjectPhoenix/4.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: Mapping[str, Any], timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ProjectPhoenix/4.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TTLCache:
    """Tiny time-to-live cache: reload after ``ttl_ns``; keep serving the last good value on failure."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[int, Any]] = {}
        self._last_error: dict[str, str] = {}

    def get(self, key: str, ttl_ns: int, loader: Callable[[], Any], now_ns: int | None = None) -> Any:
        now = int(time.time_ns()) if now_ns is None else int(now_ns)
        entry = self._entries.get(key)
        if entry is not None and now - entry[0] < ttl_ns:
            return entry[1]
        try:
            value = loader()
        except Exception as exc:
            self._last_error[key] = str(exc)
            logger.warning("Feed '%s' refresh failed (%s); %s", key, exc, "serving stale value" if entry else "no value")
            return entry[1] if entry is not None else None
        self._entries[key] = (now, value)
        self._last_error.pop(key, None)
        return value

    def age_ns(self, key: str, now_ns: int | None = None) -> int | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        now = int(time.time_ns()) if now_ns is None else int(now_ns)
        return now - entry[0]

    def last_error(self, key: str) -> str | None:
        return self._last_error.get(key)

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)


# ------------------------------------------------------------------------------------------------
# Hyperliquid auxiliary endpoints
# ------------------------------------------------------------------------------------------------


class PredictedFunding(BaseModel):
    """One venue's predicted funding for an asset, with the per-hour and per-day normalisation."""

    model_config = ConfigDict(frozen=True)

    venue: str
    funding_rate: float = Field(description="Rate for one funding interval, as a decimal")
    interval_hours: float = Field(gt=0.0)
    next_funding_ms: int = Field(ge=0)

    @property
    def rate_per_hour(self) -> float:
        return self.funding_rate / self.interval_hours

    @property
    def rate_per_day(self) -> float:
        return self.rate_per_hour * 24.0


def parse_predicted_fundings(payload: Any) -> dict[str, dict[str, PredictedFunding]]:
    """``[[coin, [[venue, {fundingRate, nextFundingTime, fundingIntervalHours}], ...]], ...]`` -> nested dict."""
    out: dict[str, dict[str, PredictedFunding]] = {}
    if not isinstance(payload, list):
        return out
    for item in payload:
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[1], list)):
            continue
        coin = str(item[0]).upper()
        venues: dict[str, PredictedFunding] = {}
        for venue_row in item[1]:
            if not (isinstance(venue_row, list) and len(venue_row) == 2 and isinstance(venue_row[1], dict)):
                continue
            info = venue_row[1]
            try:
                venues[str(venue_row[0])] = PredictedFunding(
                    venue=str(venue_row[0]),
                    funding_rate=float(info.get("fundingRate", 0.0)),
                    interval_hours=float(info.get("fundingIntervalHours", 1) or 1),
                    next_funding_ms=int(info.get("nextFundingTime", 0) or 0),
                )
            except (TypeError, ValueError):
                continue
        if venues:
            out[coin] = venues
    return out


def fetch_predicted_fundings(url: str = HYPERLIQUID_INFO_URL, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> dict[str, dict[str, PredictedFunding]]:
    return parse_predicted_fundings(_post_json(url, {"type": "predictedFundings"}, timeout_sec))


def parse_oi_cap_assets(payload: Any) -> frozenset[str]:
    """``["COIN", ...]`` -> set of assets currently at their open-interest cap."""
    if not isinstance(payload, list):
        return frozenset()
    return frozenset(str(x).upper() for x in payload if isinstance(x, str))


def fetch_perps_at_oi_cap(url: str = HYPERLIQUID_INFO_URL, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> frozenset[str]:
    return parse_oi_cap_assets(_post_json(url, {"type": "perpsAtOpenInterestCap"}, timeout_sec))


class PerpDexLimits(BaseModel):
    """Open-interest limits of a builder-deployed perp dex (``None`` for the main dex, which has none)."""

    model_config = ConfigDict(frozen=True)

    dex: str
    total_oi_cap_usd: float | None = None
    oi_size_cap_per_perp: float | None = None
    max_transfer_notional_usd: float | None = None
    asset_streaming_oi_caps_usd: dict[str, float] = Field(default_factory=dict)


def parse_perp_dex_limits(dex: str, payload: Any) -> PerpDexLimits | None:
    if not isinstance(payload, dict):
        return None

    def _f(key: str) -> float | None:
        v = payload.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    caps: dict[str, float] = {}
    for row in payload.get("coinToOiCap", payload.get("assetToStreamingOiCap", [])) or []:
        if isinstance(row, list) and len(row) == 2:
            try:
                caps[str(row[0])] = float(row[1])
            except (TypeError, ValueError):
                continue
    return PerpDexLimits(
        dex=dex,
        total_oi_cap_usd=_f("totalOiCap"),
        oi_size_cap_per_perp=_f("oiSzCapPerPerp"),
        max_transfer_notional_usd=_f("maxTransferNtl"),
        asset_streaming_oi_caps_usd=caps,
    )


def fetch_perp_dex_limits(dex: str, url: str = HYPERLIQUID_INFO_URL, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> PerpDexLimits | None:
    return parse_perp_dex_limits(dex, _post_json(url, {"type": "perpDexLimits", "dex": dex}, timeout_sec))


# ------------------------------------------------------------------------------------------------
# Stablecoin marks
# ------------------------------------------------------------------------------------------------


class StablecoinQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    venue: str
    price: float = Field(gt=0.0)
    quote_currency: Literal["USD", "USDT"] = "USD"


class StablecoinMarks(BaseModel):
    """Per-symbol medians across venues and the resulting peg deviation."""

    model_config = ConfigDict(frozen=True)

    quotes: list[StablecoinQuote]
    median_usd: dict[str, float]
    venues_per_symbol: dict[str, int]
    depeg_bps: dict[str, float]
    max_depeg_bps: float | None = Field(default=None, description="Largest deviation among symbols quoted by two or more venues")
    ts_ns: int = 0

    @property
    def symbols_with_two_venues(self) -> list[str]:
        return sorted(s for s, n in self.venues_per_symbol.items() if n >= 2)


def _mid(bid: Any, ask: Any) -> float:
    b, a = float(bid), float(ask)
    if b > 0.0 and a > 0.0:
        return (b + a) / 2.0
    return b if b > 0.0 else a


def _parse_coinbase_spot(d: Any) -> float:
    return float(d["data"]["amount"])


def _parse_kraken(d: Any) -> float:
    res = d["result"]
    first = next(iter(res.values()))
    return _mid(first["b"][0], first["a"][0])


def _parse_bitstamp(d: Any) -> float:
    return _mid(d["bid"], d["ask"])


def _parse_gemini(d: Any) -> float:
    return _mid(d["bid"], d["ask"])


def _parse_okx_ticker(d: Any) -> float:
    row = d["data"][0]
    return _mid(row["bidPx"], row["askPx"])


def _parse_kucoin_level1(d: Any) -> float:
    row = d["data"]
    return _mid(row["bestBid"], row["bestAsk"])


class StablecoinSource(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbol: str
    venue: str
    url: str
    quote_currency: Literal["USD", "USDT"]
    parser_name: str


STABLECOIN_PARSERS: Final[dict[str, Callable[[Any], float]]] = {
    "coinbase_spot": _parse_coinbase_spot,
    "kraken": _parse_kraken,
    "bitstamp": _parse_bitstamp,
    "gemini": _parse_gemini,
    "okx_ticker": _parse_okx_ticker,
    "kucoin_level1": _parse_kucoin_level1,
}

STABLECOIN_SOURCES: Final[tuple[StablecoinSource, ...]] = (
    StablecoinSource(symbol="USDT", venue="coinbase", url="https://api.coinbase.com/v2/prices/USDT-USD/spot", quote_currency="USD", parser_name="coinbase_spot"),
    StablecoinSource(symbol="USDT", venue="kraken", url="https://api.kraken.com/0/public/Ticker?pair=USDTZUSD", quote_currency="USD", parser_name="kraken"),
    StablecoinSource(symbol="USDT", venue="bitstamp", url="https://www.bitstamp.net/api/v2/ticker/usdtusd/", quote_currency="USD", parser_name="bitstamp"),
    StablecoinSource(symbol="USDT", venue="gemini", url="https://api.gemini.com/v1/pubticker/usdtusd", quote_currency="USD", parser_name="gemini"),
    StablecoinSource(symbol="USDC", venue="kraken", url="https://api.kraken.com/0/public/Ticker?pair=USDCUSD", quote_currency="USD", parser_name="kraken"),
    StablecoinSource(symbol="USDC", venue="bitstamp", url="https://www.bitstamp.net/api/v2/ticker/usdcusd/", quote_currency="USD", parser_name="bitstamp"),
    StablecoinSource(symbol="USDC", venue="gemini", url="https://api.gemini.com/v1/pubticker/usdcusd", quote_currency="USD", parser_name="gemini"),
    StablecoinSource(symbol="USDC", venue="okx", url="https://www.okx.com/api/v5/market/ticker?instId=USDC-USDT", quote_currency="USDT", parser_name="okx_ticker"),
    StablecoinSource(symbol="USDE", venue="kraken", url="https://api.kraken.com/0/public/Ticker?pair=USDEUSD", quote_currency="USD", parser_name="kraken"),
    StablecoinSource(symbol="USDE", venue="kucoin", url="https://api.kucoin.com/api/v1/market/orderbook/level1?symbol=USDE-USDT", quote_currency="USDT", parser_name="kucoin_level1"),
)


def compute_stablecoin_marks(quotes: Sequence[StablecoinQuote], ts_ns: int = 0) -> StablecoinMarks:
    """Median USD mark per symbol; USDT-quoted prices are converted with the USDT/USD median.

    A symbol enters ``max_depeg_bps`` only when at least two venues quote it. If no USD quote for
    USDT exists, USDT-quoted prices cannot be converted and are dropped rather than assumed at par.
    """
    usd_quotes: dict[str, list[float]] = {}
    for q in quotes:
        if q.quote_currency == "USD":
            usd_quotes.setdefault(q.symbol.upper(), []).append(q.price)
    usdt_usd = statistics.median(usd_quotes["USDT"]) if usd_quotes.get("USDT") else None
    for q in quotes:
        if q.quote_currency == "USDT" and usdt_usd is not None:
            usd_quotes.setdefault(q.symbol.upper(), []).append(q.price * usdt_usd)

    median_usd = {s: float(statistics.median(v)) for s, v in usd_quotes.items() if v}
    venues = {s: len(v) for s, v in usd_quotes.items()}
    depeg = {s: abs(m - 1.0) * 10_000.0 for s, m in median_usd.items()}
    eligible = [depeg[s] for s, n in venues.items() if n >= 2]
    return StablecoinMarks(
        quotes=list(quotes),
        median_usd=median_usd,
        venues_per_symbol=venues,
        depeg_bps=depeg,
        max_depeg_bps=max(eligible) if eligible else None,
        ts_ns=ts_ns,
    )


def fetch_stablecoin_quotes(
    sources: Sequence[StablecoinSource] = STABLECOIN_SOURCES,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> list[StablecoinQuote]:
    """Fetch every source in parallel; a venue that fails is simply absent."""

    def _one(src: StablecoinSource) -> StablecoinQuote | None:
        try:
            price = STABLECOIN_PARSERS[src.parser_name](_get_json(src.url, timeout_sec=timeout_sec))
            if not math.isfinite(price) or price <= 0.0:
                return None
            return StablecoinQuote(symbol=src.symbol, venue=src.venue, price=price, quote_currency=src.quote_currency)
        except Exception as exc:
            logger.debug("Stablecoin quote %s@%s failed: %s", src.symbol, src.venue, exc)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as pool:
        results = list(pool.map(_one, sources))
    return [q for q in results if q is not None]


# ------------------------------------------------------------------------------------------------
# Deribit
# ------------------------------------------------------------------------------------------------


class DvolSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    index: float = Field(ge=0.0)
    zscore: float
    n_obs: int = Field(ge=1)
    window_hours: int = Field(ge=1)
    ts_ms: int = Field(ge=0)


def parse_dvol_series(payload: Any) -> list[tuple[int, float]]:
    """``{"result": {"data": [[ts, open, high, low, close], ...]}}`` -> ``(ts_ms, close)`` oldest first."""
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    rows = result.get("data", []) if isinstance(result, dict) else []
    out: list[tuple[int, float]] = []
    for r in rows:
        if isinstance(r, list) and len(r) >= 5:
            try:
                out.append((int(r[0]), float(r[4])))
            except (TypeError, ValueError):
                continue
    out.sort(key=lambda x: x[0])
    return out


def compute_dvol_snapshot(series: Sequence[tuple[int, float]], currency: str, window_hours: int = DVOL_LOOKBACK_HOURS) -> DvolSnapshot | None:
    """Latest DVOL close and its z-score against the trailing window (excluding the latest point)."""
    if not series:
        return None
    closes = [c for _, c in series]
    latest_ts, latest = series[-1]
    trailing = closes[:-1][-window_hours:]
    if len(trailing) >= 2:
        mean = statistics.fmean(trailing)
        std = statistics.pstdev(trailing)
        z = (latest - mean) / std if std > 1e-9 else 0.0
    else:
        z = 0.0
    return DvolSnapshot(currency=currency.upper(), index=float(latest), zscore=float(z), n_obs=len(closes), window_hours=window_hours, ts_ms=int(latest_ts))


def fetch_dvol(
    currency: str = "BTC",
    lookback_hours: int = DVOL_LOOKBACK_HOURS,
    resolution_sec: int = 3600,
    base_url: str = DERIBIT_BASE_URL,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> DvolSnapshot | None:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (lookback_hours + 2) * 3_600_000
    payload = _get_json(
        f"{base_url}/get_volatility_index_data",
        {"currency": currency.upper(), "resolution": resolution_sec, "start_timestamp": start_ms, "end_timestamp": end_ms},
        timeout_sec=timeout_sec,
    )
    return compute_dvol_snapshot(parse_dvol_series(payload), currency, window_hours=lookback_hours)


class DeribitInstrument(BaseModel):
    model_config = ConfigDict(frozen=True)

    instrument_name: str
    expiration_ms: int = Field(ge=0)
    strike: float = Field(gt=0.0)
    option_type: Literal["call", "put"]
    settlement_period: str = ""

    def days_to_expiry(self, now_ms: int) -> float:
        return (self.expiration_ms - now_ms) / 86_400_000.0


def parse_instruments(payload: Any) -> list[DeribitInstrument]:
    result = payload.get("result", payload) if isinstance(payload, dict) else payload
    out: list[DeribitInstrument] = []
    for row in result if isinstance(result, list) else []:
        if not isinstance(row, dict) or row.get("option_type") not in ("call", "put"):
            continue
        try:
            out.append(
                DeribitInstrument(
                    instrument_name=str(row["instrument_name"]),
                    expiration_ms=int(row["expiration_timestamp"]),
                    strike=float(row["strike"]),
                    option_type=row["option_type"],
                    settlement_period=str(row.get("settlement_period", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda i: (i.expiration_ms, i.strike, i.option_type))
    return out


def fetch_instruments(currency: str = "BTC", base_url: str = DERIBIT_BASE_URL, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> list[DeribitInstrument]:
    return parse_instruments(_get_json(f"{base_url}/get_instruments", {"currency": currency.upper(), "kind": "option", "expired": "false"}, timeout_sec))


def instruments_for_tenor(instruments: Sequence[DeribitInstrument], now_ms: int, target_days: float, tolerance_days: float = 7.0) -> list[DeribitInstrument]:
    """Instruments of the expiry closest to ``target_days`` ahead (within tolerance), all strikes."""
    candidates = [i for i in instruments if abs(i.days_to_expiry(now_ms) - target_days) <= tolerance_days]
    if not candidates:
        return []
    best_expiry = min(candidates, key=lambda i: abs(i.days_to_expiry(now_ms) - target_days)).expiration_ms
    return [i for i in candidates if i.expiration_ms == best_expiry]


class DeribitOptionTicker(BaseModel):
    model_config = ConfigDict(frozen=True)

    instrument_name: str
    mark_iv: float = Field(ge=0.0)
    bid_iv: float | None = None
    ask_iv: float | None = None
    mark_price: float = Field(ge=0.0, description="In units of the underlying")
    underlying_price: float = Field(gt=0.0)
    index_price: float = Field(gt=0.0)
    open_interest: float = Field(ge=0.0)
    delta: float
    gamma: float
    vega: float
    theta: float
    timestamp_ms: int = Field(ge=0)


def parse_option_ticker(payload: Any) -> DeribitOptionTicker | None:
    result = payload.get("result", payload) if isinstance(payload, dict) else None
    if not isinstance(result, dict) or "greeks" not in result:
        return None
    g = result.get("greeks", {})
    try:
        return DeribitOptionTicker(
            instrument_name=str(result["instrument_name"]) if "instrument_name" in result else str(result.get("underlying_index", "")),
            mark_iv=float(result.get("mark_iv", 0.0)),
            bid_iv=float(result["bid_iv"]) if result.get("bid_iv") is not None else None,
            ask_iv=float(result["ask_iv"]) if result.get("ask_iv") is not None else None,
            mark_price=float(result.get("mark_price", 0.0)),
            underlying_price=float(result["underlying_price"]),
            index_price=float(result.get("index_price") or result["underlying_price"]),
            open_interest=float(result.get("open_interest", 0.0)),
            delta=float(g.get("delta", 0.0)),
            gamma=float(g.get("gamma", 0.0)),
            vega=float(g.get("vega", 0.0)),
            theta=float(g.get("theta", 0.0)),
            timestamp_ms=int(result.get("timestamp", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def fetch_option_ticker(instrument_name: str, base_url: str = DERIBIT_BASE_URL, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> DeribitOptionTicker | None:
    payload = _get_json(f"{base_url}/ticker", {"instrument_name": instrument_name}, timeout_sec)
    ticker = parse_option_ticker(payload)
    if ticker is not None and not ticker.instrument_name:
        ticker = ticker.model_copy(update={"instrument_name": instrument_name})
    return ticker


# ------------------------------------------------------------------------------------------------
# Macro inputs for the risk committee (P0.8)
# ------------------------------------------------------------------------------------------------


class MacroInputs(BaseModel):
    """What the macro sentinel receives. ``None`` means the feed was unavailable."""

    model_config = ConfigDict(frozen=True)

    ts_ns: int = 0
    stablecoin_depeg_bps: float | None = None
    dvol_index: float | None = None
    dvol_zscore: float | None = None
    stablecoin_marks: StablecoinMarks | None = None
    dvol: DvolSnapshot | None = None

    @property
    def stablecoin_marks_available(self) -> bool:
        return self.stablecoin_depeg_bps is not None

    @property
    def dvol_available(self) -> bool:
        return self.dvol_index is not None


class MacroInputsFeed(Protocol):
    def macro_inputs(self, ts_ns: int) -> MacroInputs: ...


class ScriptedMacroInputsFeed:
    """Fixed macro inputs for tests and replays."""

    def __init__(self, inputs: MacroInputs | None = None) -> None:
        self._inputs = inputs or MacroInputs()

    def macro_inputs(self, ts_ns: int) -> MacroInputs:
        return self._inputs.model_copy(update={"ts_ns": ts_ns})


class LiveMacroInputsFeed:
    """Stablecoin marks and BTC DVOL with a TTL; failures leave fields ``None``."""

    def __init__(self, ttl_ns: int = DEFAULT_TTL_NS, timeout_sec: float = DEFAULT_TIMEOUT_SEC, dvol_currency: str = "BTC") -> None:
        self._cache = TTLCache()
        self._ttl_ns = ttl_ns
        self._timeout = timeout_sec
        self._dvol_currency = dvol_currency

    def macro_inputs(self, ts_ns: int) -> MacroInputs:
        marks_raw = self._cache.get(
            "stablecoin_marks",
            self._ttl_ns,
            lambda: compute_stablecoin_marks(fetch_stablecoin_quotes(timeout_sec=self._timeout), ts_ns=ts_ns),
            now_ns=ts_ns,
        )
        dvol_raw = self._cache.get("dvol", self._ttl_ns, lambda: fetch_dvol(self._dvol_currency, timeout_sec=self._timeout), now_ns=ts_ns)
        marks = marks_raw if isinstance(marks_raw, StablecoinMarks) else None
        dvol = dvol_raw if isinstance(dvol_raw, DvolSnapshot) else None
        return MacroInputs(
            ts_ns=ts_ns,
            stablecoin_depeg_bps=marks.max_depeg_bps if marks is not None else None,
            dvol_index=dvol.index if dvol is not None else None,
            dvol_zscore=dvol.zscore if dvol is not None else None,
            stablecoin_marks=marks,
            dvol=dvol,
        )


# ------------------------------------------------------------------------------------------------
# Hyperliquid auxiliary feed (predicted fundings, OI caps)
# ------------------------------------------------------------------------------------------------


class HyperliquidAuxSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    ts_ns: int = 0
    predicted_fundings: dict[str, dict[str, PredictedFunding]] = Field(default_factory=dict)
    oi_cap_assets: frozenset[str] = Field(default_factory=frozenset)
    predicted_fundings_available: bool = False
    oi_caps_available: bool = False


class HyperliquidAuxFeed(Protocol):
    def aux_snapshot(self, ts_ns: int, coins: Sequence[str]) -> HyperliquidAuxSnapshot: ...


class ScriptedHyperliquidAuxFeed:
    def __init__(self, snapshot: HyperliquidAuxSnapshot | None = None) -> None:
        self._snap = snapshot or HyperliquidAuxSnapshot()

    def aux_snapshot(self, ts_ns: int, coins: Sequence[str]) -> HyperliquidAuxSnapshot:
        return self._snap.model_copy(update={"ts_ns": ts_ns})


class LiveHyperliquidAuxFeed:
    def __init__(self, ttl_ns: int = DEFAULT_TTL_NS, timeout_sec: float = DEFAULT_TIMEOUT_SEC, url: str = HYPERLIQUID_INFO_URL) -> None:
        self._cache = TTLCache()
        self._ttl_ns = ttl_ns
        self._timeout = timeout_sec
        self._url = url

    def aux_snapshot(self, ts_ns: int, coins: Sequence[str]) -> HyperliquidAuxSnapshot:
        pf_raw = self._cache.get("predicted_fundings", self._ttl_ns, lambda: fetch_predicted_fundings(self._url, self._timeout), now_ns=ts_ns)
        caps_raw = self._cache.get("oi_caps", self._ttl_ns, lambda: fetch_perps_at_oi_cap(self._url, self._timeout), now_ns=ts_ns)
        wanted = {c.upper() for c in coins}
        pf = {k: v for k, v in pf_raw.items() if k in wanted} if isinstance(pf_raw, dict) else {}
        caps = caps_raw if isinstance(caps_raw, frozenset) else frozenset()
        return HyperliquidAuxSnapshot(
            ts_ns=ts_ns,
            predicted_fundings=pf,
            oi_cap_assets=caps,
            predicted_fundings_available=isinstance(pf_raw, dict) and bool(pf_raw),
            oi_caps_available=isinstance(caps_raw, frozenset),
        )


__all__ = [
    "DEFAULT_TTL_NS",
    "DeribitInstrument",
    "DeribitOptionTicker",
    "DvolSnapshot",
    "HyperliquidAuxFeed",
    "HyperliquidAuxSnapshot",
    "LiveHyperliquidAuxFeed",
    "LiveMacroInputsFeed",
    "MacroInputs",
    "MacroInputsFeed",
    "PerpDexLimits",
    "PredictedFunding",
    "STABLECOIN_SOURCES",
    "ScriptedHyperliquidAuxFeed",
    "ScriptedMacroInputsFeed",
    "StablecoinMarks",
    "StablecoinQuote",
    "StablecoinSource",
    "TTLCache",
    "compute_dvol_snapshot",
    "compute_stablecoin_marks",
    "fetch_dvol",
    "fetch_instruments",
    "fetch_option_ticker",
    "fetch_perp_dex_limits",
    "fetch_perps_at_oi_cap",
    "fetch_predicted_fundings",
    "fetch_stablecoin_quotes",
    "instruments_for_tenor",
    "parse_dvol_series",
    "parse_instruments",
    "parse_oi_cap_assets",
    "parse_option_ticker",
    "parse_perp_dex_limits",
    "parse_predicted_fundings",
]
