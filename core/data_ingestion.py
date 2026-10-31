"""Real-Time Multi-Asset Market Data Ingestion Engine for Hyperliquid L1 & Bybit V5.

Streams simultaneous BTC, ETH, and SOL mid prices, mark prices, spot oracle prices,
Open Interest, 1-hour funding rates, and top-20 L2 order book depth from Hyperliquid L1
(``https://api.hyperliquid.xyz/info``) with Bybit V5 linear perps as an automatic failover.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, Protocol
import urllib.error
import urllib.request

import httpx

from core.contracts import (
    AssetMarketSnapshot,
    DataAvailability,
    DollarBar,
    L2Level,
    L2OrderBookSnapshot,
    MultiAssetMarketSnapshot,
)

logger = logging.getLogger("phoenix.data_ingestion")

HYPERLIQUID_INFO_URL: Final[str] = "https://api.hyperliquid.xyz/info"
BYBIT_TICKERS_URL: Final[str] = "https://api.bybit.com/v5/market/tickers"
BYBIT_ORDERBOOK_URL: Final[str] = "https://api.bybit.com/v5/market/orderbook"

DEFAULT_ASSETS: Final[tuple[str, ...]] = ("BTC", "ETH", "SOL")


def normalize_coin_symbol(symbol: str) -> str:
    """Normalize exchange symbol variations (e.g. BTCUSDT, BTCUSDT-PERP, btc) to base coin (BTC)."""
    clean = (
        symbol.strip()
        .upper()
        .removesuffix("-PERP")
        .removesuffix("USDT-PERP")
        .removesuffix("USDC-PERP")
        .removesuffix("USDT")
        .removesuffix("USDC")
        .removesuffix("USD")
    )
    return clean or symbol.strip().upper()


def _compute_l2_depth_and_spread(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Compute (spread_bps, depth_10bps_usd, depth_50bps_usd) from sorted bids and asks."""
    if not bids or not asks:
        return 0.0, 0.0, 0.0

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0.0:
        return 0.0, 0.0, 0.0

    spread_bps = max(0.0, (best_ask - best_bid) / mid * 10_000.0)

    thresh_10bps_bid = mid * (1.0 - 0.0010)
    thresh_10bps_ask = mid * (1.0 + 0.0010)
    thresh_50bps_bid = mid * (1.0 - 0.0050)
    thresh_50bps_ask = mid * (1.0 + 0.0050)

    depth_10bps = sum(p * sz for p, sz in bids if p >= thresh_10bps_bid) + sum(
        p * sz for p, sz in asks if p <= thresh_10bps_ask
    )
    depth_50bps = sum(p * sz for p, sz in bids if p >= thresh_50bps_bid) + sum(
        p * sz for p, sz in asks if p <= thresh_50bps_ask
    )

    return round(spread_bps, 3), round(depth_10bps, 2), round(depth_50bps, 2)


class LiveMarketDataEngine:
    """Sub-second Live Multi-Asset Market Data Ingestion Engine.

    Features:
    1. Simultaneous BTC, ETH, and SOL context via Hyperliquid L1 ``metaAndAssetCtxs``.
    2. Top-20 L2 order book depth streaming via Hyperliquid L1 ``l2Book``.
    3. Automatic sub-second failover to Bybit V5 Linear Perps when Hyperliquid is degraded.
    4. Deterministic simulated fallback if all external networks are completely offline.
    5. Clean asynchronous and synchronous interfaces for both batch and continuous polling.
    """

    def __init__(
        self,
        timeout_sec: float = 1.0,
        hl_url: str = HYPERLIQUID_INFO_URL,
        bybit_tickers_url: str = BYBIT_TICKERS_URL,
        bybit_book_url: str = BYBIT_ORDERBOOK_URL,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.hl_url = hl_url
        self.bybit_tickers_url = bybit_tickers_url
        self.bybit_book_url = bybit_book_url

        self._cached_snapshot: MultiAssetMarketSnapshot | None = None
        self._last_success_source: str = "INITIALIZING"
        self._last_success_ts_ns: int = 0
        self._background_task: asyncio.Task[None] | None = None
        self._running: bool = False

    # -------------------------------------------------------------------------
    # Hyperliquid L1 Primary Ingestion (Async)
    # -------------------------------------------------------------------------

    async def fetch_hyperliquid_meta_and_contexts(
        self, client: httpx.AsyncClient | None = None
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Fetch Hyperliquid L1 universe metadata and asset contexts in a single request."""
        payload = {"type": "metaAndAssetCtxs"}
        headers = {"Content-Type": "application/json", "User-Agent": "ProjectPhoenix/4.0"}

        if client is not None:
            resp = await client.post(self.hl_url, json=payload, headers=headers, timeout=self.timeout_sec)
            data = resp.json()
        else:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as c:
                resp = await c.post(self.hl_url, json=payload, headers=headers)
                data = resp.json()

        universe = data[0]["universe"]
        contexts = data[1]
        name_to_idx: dict[str, int] = {item["name"]: i for i, item in enumerate(universe)}
        return name_to_idx, contexts

    async def fetch_hyperliquid_l2_book(
        self, coin: str, client: httpx.AsyncClient | None = None
    ) -> L2OrderBookSnapshot:
        """Fetch top-20 bid/ask levels for a coin from Hyperliquid L1 ``l2Book``."""
        norm_coin = normalize_coin_symbol(coin)
        payload = {"type": "l2Book", "coin": norm_coin}
        headers = {"Content-Type": "application/json", "User-Agent": "ProjectPhoenix/4.0"}

        if client is not None:
            resp = await client.post(self.hl_url, json=payload, headers=headers, timeout=self.timeout_sec)
            data = resp.json()
        else:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as c:
                resp = await c.post(self.hl_url, json=payload, headers=headers)
                data = resp.json()

        time_ms = int(data.get("time", time.time() * 1000))
        raw_bids = data.get("levels", [[], []])[0][:20]
        raw_asks = data.get("levels", [[], []])[1][:20]

        bids: list[tuple[float, float]] = [(float(lvl["px"]), float(lvl["sz"])) for lvl in raw_bids]
        asks: list[tuple[float, float]] = [(float(lvl["px"]), float(lvl["sz"])) for lvl in raw_asks]
        levels_bids = [
            L2Level(price=float(lvl["px"]), size=float(lvl["sz"]), order_count=int(lvl.get("n", 1)))
            for lvl in raw_bids
        ]
        levels_asks = [
            L2Level(price=float(lvl["px"]), size=float(lvl["sz"]), order_count=int(lvl.get("n", 1)))
            for lvl in raw_asks
        ]

        spread_bps, depth_10bps, depth_50bps = _compute_l2_depth_and_spread(bids, asks)

        return L2OrderBookSnapshot(
            instrument_id=norm_coin,
            bids=bids,
            asks=asks,
            levels_bids=levels_bids,
            levels_asks=levels_asks,
            spread_bps=spread_bps,
            depth_10bps_usd=depth_10bps,
            depth_50bps_usd=depth_50bps,
            timestamp_ns=time_ms * 1_000_000,
        )

    async def fetch_hyperliquid_multi_asset_snapshot(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        include_l2: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> MultiAssetMarketSnapshot:
        """Fetch simultaneous multi-asset context and L2 books from Hyperliquid L1."""
        now_ns = time.time_ns()

        if client is not None:
            name_to_idx, contexts = await self.fetch_hyperliquid_meta_and_contexts(client)
            if include_l2:
                l2_tasks = [self.fetch_hyperliquid_l2_book(coin, client=client) for coin in coins]
                l2_books = await asyncio.gather(*l2_tasks)
            else:
                l2_books = [L2OrderBookSnapshot(instrument_id=normalize_coin_symbol(c)) for c in coins]
        else:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as c:
                name_to_idx, contexts = await self.fetch_hyperliquid_meta_and_contexts(c)
                if include_l2:
                    l2_tasks = [self.fetch_hyperliquid_l2_book(coin, client=c) for coin in coins]
                    l2_books = await asyncio.gather(*l2_tasks)
                else:
                    l2_books = [L2OrderBookSnapshot(instrument_id=normalize_coin_symbol(coin)) for coin in coins]

        coin_to_l2 = {normalize_coin_symbol(coins[i]): l2_books[i] for i in range(len(coins))}
        asset_snapshots: dict[str, AssetMarketSnapshot] = {}

        for coin in coins:
            norm = normalize_coin_symbol(coin)
            if norm not in name_to_idx:
                continue
            idx = name_to_idx[norm]
            ctx = contexts[idx]

            mark_px = float(ctx.get("markPx", 0.0))
            mid_px = float(ctx.get("midPx", mark_px)) or mark_px
            oracle_px = float(ctx.get("oraclePx", mark_px)) or mark_px
            oi = float(ctx.get("openInterest", 0.0))
            funding_1h = float(ctx.get("funding", 0.0))
            day_ntl_vol = float(ctx.get("dayNtlVlm", 0.0))
            funding_8h_equiv = funding_1h * 8.0
            annualized_apr = funding_1h * 24.0 * 365.0 * 100.0
            basis_bps = (
                round((mark_px - oracle_px) / oracle_px * 10_000.0, 3) if oracle_px > 0 else 0.0
            )

            l2_snap = coin_to_l2.get(norm, L2OrderBookSnapshot(instrument_id=norm))

            asset_snapshots[norm] = AssetMarketSnapshot(
                instrument_id=norm,
                mid_price=mid_px,
                mark_price=mark_px,
                oracle_price=oracle_px,
                open_interest=oi,
                funding_rate_1h=funding_1h,
                funding_rate_8h_equiv=funding_8h_equiv,
                day_volume_usd=day_ntl_vol,
                l2_book=l2_snap,
                annualized_apr_pct=round(annualized_apr, 4),
                basis_bps=basis_bps,
                feed_source="HYPERLIQUID-L1",
                timestamp_ns=now_ns,
            )

        if not asset_snapshots:
            raise ValueError(f"No matching assets found for {coins} in Hyperliquid metadata.")

        return MultiAssetMarketSnapshot(
            timestamp_ns=now_ns,
            assets=asset_snapshots,
            feed_source="HYPERLIQUID-L1",
        )

    # -------------------------------------------------------------------------
    # Bybit V5 Failover Ingestion (Async)
    # -------------------------------------------------------------------------

    async def fetch_bybit_ticker(
        self, coin: str, client: httpx.AsyncClient | None = None
    ) -> dict[str, Any]:
        """Fetch linear ticker for a single coin from Bybit V5."""
        norm_coin = normalize_coin_symbol(coin)
        symbol = f"{norm_coin}USDT"
        params = {"category": "linear", "symbol": symbol}
        headers = {"User-Agent": "ProjectPhoenix/4.0"}

        if client is not None:
            resp = await client.get(
                self.bybit_tickers_url, params=params, headers=headers, timeout=self.timeout_sec
            )
            data = resp.json()
        else:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as c:
                resp = await c.get(
                    self.bybit_tickers_url, params=params, headers=headers
                )
                data = resp.json()

        lst = data.get("result", {}).get("list", [])
        if not lst:
            raise ValueError(f"Empty ticker list returned by Bybit for {symbol}")
        return lst[0]  # type: ignore[no-any-return]

    async def fetch_bybit_l2_book(
        self, coin: str, client: httpx.AsyncClient | None = None
    ) -> L2OrderBookSnapshot:
        """Fetch top-20 orderbook depth for a coin from Bybit V5 linear perps."""
        norm_coin = normalize_coin_symbol(coin)
        symbol = f"{norm_coin}USDT"
        params = {"category": "linear", "symbol": symbol, "limit": "20"}
        headers = {"User-Agent": "ProjectPhoenix/4.0"}

        if client is not None:
            resp = await client.get(
                self.bybit_book_url, params=params, headers=headers, timeout=self.timeout_sec
            )
            data = resp.json()
        else:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as c:
                resp = await c.get(self.bybit_book_url, params=params, headers=headers)
                data = resp.json()

        res = data.get("result", {})
        raw_bids = res.get("b", [])[:20]
        raw_asks = res.get("a", [])[:20]
        time_ms = int(data.get("time", time.time() * 1000))

        bids: list[tuple[float, float]] = [(float(b[0]), float(b[1])) for b in raw_bids]
        asks: list[tuple[float, float]] = [(float(a[0]), float(a[1])) for a in raw_asks]
        levels_bids = [
            L2Level(price=float(b[0]), size=float(b[1]), order_count=1) for b in raw_bids
        ]
        levels_asks = [
            L2Level(price=float(a[0]), size=float(a[1]), order_count=1) for a in raw_asks
        ]

        spread_bps, depth_10bps, depth_50bps = _compute_l2_depth_and_spread(bids, asks)

        return L2OrderBookSnapshot(
            instrument_id=norm_coin,
            bids=bids,
            asks=asks,
            levels_bids=levels_bids,
            levels_asks=levels_asks,
            spread_bps=spread_bps,
            depth_10bps_usd=depth_10bps,
            depth_50bps_usd=depth_50bps,
            timestamp_ns=time_ms * 1_000_000,
        )

    async def fetch_bybit_multi_asset_snapshot(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        include_l2: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> MultiAssetMarketSnapshot:
        """Fetch simultaneous multi-asset context and L2 books from Bybit V5 Linear."""
        now_ns = time.time_ns()

        async def _fetch_all(c: httpx.AsyncClient) -> tuple[list[dict[str, Any]], list[L2OrderBookSnapshot]]:
            ticker_tasks = [self.fetch_bybit_ticker(coin, client=c) for coin in coins]
            if include_l2:
                book_tasks = [self.fetch_bybit_l2_book(coin, client=c) for coin in coins]
                results = await asyncio.gather(
                    asyncio.gather(*ticker_tasks),
                    asyncio.gather(*book_tasks),
                )
                return results[0], results[1]
            tickers = await asyncio.gather(*ticker_tasks)
            empty_books = [L2OrderBookSnapshot(instrument_id=normalize_coin_symbol(coin)) for coin in coins]
            return tickers, empty_books

        if client is not None:
            raw_tickers, l2_books = await _fetch_all(client)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as c:
                raw_tickers, l2_books = await _fetch_all(c)

        coin_to_l2 = {normalize_coin_symbol(coins[i]): l2_books[i] for i in range(len(coins))}
        asset_snapshots: dict[str, AssetMarketSnapshot] = {}

        for i, coin in enumerate(coins):
            norm = normalize_coin_symbol(coin)
            item = raw_tickers[i]

            last_px = float(item.get("lastPrice", 0.0))
            mark_px = float(item.get("markPrice", last_px))
            index_px = float(item.get("indexPrice", mark_px))
            mid_px = last_px or mark_px
            oi = float(item.get("openInterest", 0.0))
            turnover_24h = float(item.get("turnover24h", 0.0))

            # Bybit funding rate is settled every 8 hours
            funding_8h = float(item.get("fundingRate", 0.0))
            funding_1h = funding_8h / 8.0
            annualized_apr = funding_1h * 24.0 * 365.0 * 100.0
            basis_bps = (
                round((mark_px - index_px) / index_px * 10_000.0, 3) if index_px > 0 else 0.0
            )

            l2_snap = coin_to_l2.get(norm, L2OrderBookSnapshot(instrument_id=norm))

            asset_snapshots[norm] = AssetMarketSnapshot(
                instrument_id=norm,
                mid_price=mid_px,
                mark_price=mark_px,
                oracle_price=index_px,
                open_interest=oi,
                funding_rate_1h=funding_1h,
                funding_rate_8h_equiv=funding_8h,
                day_volume_usd=turnover_24h,
                l2_book=l2_snap,
                annualized_apr_pct=round(annualized_apr, 4),
                basis_bps=basis_bps,
                feed_source="BYBIT-V5-FALLBACK",
                timestamp_ns=now_ns,
            )

        return MultiAssetMarketSnapshot(
            timestamp_ns=now_ns,
            assets=asset_snapshots,
            feed_source="BYBIT-V5-FALLBACK",
        )

    # -------------------------------------------------------------------------
    # Offline Simulation Fallback
    # -------------------------------------------------------------------------

    def create_simulated_snapshot(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        base_btc_price: float = 77_450.0,
    ) -> MultiAssetMarketSnapshot:
        """Deterministic simulated fallback if all external networks are completely unreachable."""
        now_ns = time.time_ns()
        base_prices = {"BTC": base_btc_price, "ETH": 2_244.0, "SOL": 96.65}
        asset_snapshots: dict[str, AssetMarketSnapshot] = {}

        for coin in coins:
            norm = normalize_coin_symbol(coin)
            p = base_prices.get(norm, 100.0)
            mark = round(p * 1.0001, 2)
            oracle = p
            mid = p
            oi = {"BTC": 42_500.0, "ETH": 990_000.0, "SOL": 5_300_000.0}.get(norm, 10_000.0)
            f1 = 0.0000125  # 1.25 bps per hour
            f8 = f1 * 8.0

            # Generate synthetic 20 levels
            step = p * 0.0001
            bids = [(round(p - (i + 1) * step, 2), round(1.0 + i * 0.1, 4)) for i in range(20)]
            asks = [(round(p + (i + 1) * step, 2), round(1.0 + i * 0.1, 4)) for i in range(20)]
            levels_bids = [L2Level(price=b[0], size=b[1], order_count=2) for b in bids]
            levels_asks = [L2Level(price=a[0], size=a[1], order_count=2) for a in asks]
            spread_bps, depth_10, depth_50 = _compute_l2_depth_and_spread(bids, asks)

            l2_snap = L2OrderBookSnapshot(
                instrument_id=norm,
                bids=bids,
                asks=asks,
                levels_bids=levels_bids,
                levels_asks=levels_asks,
                spread_bps=spread_bps,
                depth_10bps_usd=depth_10,
                depth_50bps_usd=depth_50,
                timestamp_ns=now_ns,
            )

            asset_snapshots[norm] = AssetMarketSnapshot(
                instrument_id=norm,
                mid_price=mid,
                mark_price=mark,
                oracle_price=oracle,
                open_interest=oi,
                funding_rate_1h=f1,
                funding_rate_8h_equiv=f8,
                day_volume_usd=round(p * oi * 0.4, 2),
                l2_book=l2_snap,
                annualized_apr_pct=round(f1 * 24.0 * 365.0 * 100.0, 4),
                basis_bps=1.0,
                feed_source="SIMULATED-FALLBACK",
                timestamp_ns=now_ns,
            )

        return MultiAssetMarketSnapshot(
            timestamp_ns=now_ns,
            assets=asset_snapshots,
            feed_source="SIMULATED-FALLBACK",
        )

    # -------------------------------------------------------------------------
    # Unified Ingestion Interface (Async & Sync)
    # -------------------------------------------------------------------------

    async def get_multi_asset_snapshot(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        include_l2: bool = True,
    ) -> MultiAssetMarketSnapshot:
        """Fetch live multi-asset snapshot: Hyperliquid L1 primary -> Bybit V5 fallback -> Simulated fallback."""
        # 1. Try Hyperliquid L1 Primary
        try:
            snapshot = await self.fetch_hyperliquid_multi_asset_snapshot(
                coins=coins, include_l2=include_l2
            )
            self._cached_snapshot = snapshot
            self._last_success_source = "HYPERLIQUID-L1"
            self._last_success_ts_ns = snapshot.timestamp_ns
            return snapshot
        except Exception as hl_exc:
            logger.warning("Hyperliquid L1 ingestion failed (%s), attempting Bybit V5 failover.", hl_exc)

        # 2. Try Bybit V5 Linear Fallback
        try:
            snapshot = await self.fetch_bybit_multi_asset_snapshot(
                coins=coins, include_l2=include_l2
            )
            self._cached_snapshot = snapshot
            self._last_success_source = "BYBIT-V5-FALLBACK"
            self._last_success_ts_ns = snapshot.timestamp_ns
            return snapshot
        except Exception as bybit_exc:
            logger.warning("Bybit V5 failover failed (%s), activating simulated fallback.", bybit_exc)

        # 3. Offline Simulated Fallback
        fallback = self.create_simulated_snapshot(coins=coins)
        self._cached_snapshot = fallback
        self._last_success_source = "SIMULATED-FALLBACK"
        self._last_success_ts_ns = fallback.timestamp_ns
        return fallback

    # -------------------------------------------------------------------------
    # Synchronous Polling Interfaces (Thread-Safe urllib / httpx)
    # -------------------------------------------------------------------------

    def fetch_hyperliquid_snapshot_sync(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        include_l2: bool = True,
    ) -> MultiAssetMarketSnapshot:
        """Synchronously poll Hyperliquid L1 using fast socket I/O."""
        req_meta = urllib.request.Request(
            self.hl_url,
            data=json.dumps({"type": "metaAndAssetCtxs"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "ProjectPhoenix/4.0"},
        )
        with urllib.request.urlopen(req_meta, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        universe = data[0]["universe"]
        contexts = data[1]
        name_to_idx = {item["name"]: i for i, item in enumerate(universe)}
        now_ns = time.time_ns()

        coin_to_l2: dict[str, L2OrderBookSnapshot] = {}
        if include_l2:
            def _fetch_single_l2(coin_sym: str) -> tuple[str, L2OrderBookSnapshot]:
                norm_c = normalize_coin_symbol(coin_sym)
                try:
                    req_l2 = urllib.request.Request(
                        self.hl_url,
                        data=json.dumps({"type": "l2Book", "coin": norm_c}).encode("utf-8"),
                        headers={"Content-Type": "application/json", "User-Agent": "ProjectPhoenix/4.0"},
                    )
                    with urllib.request.urlopen(req_l2, timeout=self.timeout_sec) as r_l2:
                        l2_data = json.loads(r_l2.read().decode("utf-8"))
                    raw_bids = l2_data.get("levels", [[], []])[0][:20]
                    raw_asks = l2_data.get("levels", [[], []])[1][:20]
                    time_ms = int(l2_data.get("time", time.time() * 1000))
                    bids = [(float(lvl["px"]), float(lvl["sz"])) for lvl in raw_bids]
                    asks = [(float(lvl["px"]), float(lvl["sz"])) for lvl in raw_asks]
                    levels_b = [
                        L2Level(price=float(lvl["px"]), size=float(lvl["sz"]), order_count=int(lvl.get("n", 1)))
                        for lvl in raw_bids
                    ]
                    levels_a = [
                        L2Level(price=float(lvl["px"]), size=float(lvl["sz"]), order_count=int(lvl.get("n", 1)))
                        for lvl in raw_asks
                    ]
                    sp, d10, d50 = _compute_l2_depth_and_spread(bids, asks)
                    return norm_c, L2OrderBookSnapshot(
                        instrument_id=norm_c,
                        bids=bids,
                        asks=asks,
                        levels_bids=levels_b,
                        levels_asks=levels_a,
                        spread_bps=sp,
                        depth_10bps_usd=d10,
                        depth_50bps_usd=d50,
                        timestamp_ns=time_ms * 1_000_000,
                    )
                except Exception:
                    return norm_c, L2OrderBookSnapshot(instrument_id=norm_c)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(coins)) as executor:
                results = list(executor.map(_fetch_single_l2, coins))
                coin_to_l2 = dict(results)

        asset_snapshots: dict[str, AssetMarketSnapshot] = {}
        for coin in coins:
            norm = normalize_coin_symbol(coin)
            if norm not in name_to_idx:
                continue
            idx = name_to_idx[norm]
            ctx = contexts[idx]

            mark_px = float(ctx.get("markPx", 0.0))
            mid_px = float(ctx.get("midPx", mark_px)) or mark_px
            oracle_px = float(ctx.get("oraclePx", mark_px)) or mark_px
            oi = float(ctx.get("openInterest", 0.0))
            funding_1h = float(ctx.get("funding", 0.0))
            day_ntl_vol = float(ctx.get("dayNtlVlm", 0.0))
            funding_8h_equiv = funding_1h * 8.0
            annualized_apr = funding_1h * 24.0 * 365.0 * 100.0
            basis_bps = (
                round((mark_px - oracle_px) / oracle_px * 10_000.0, 3) if oracle_px > 0 else 0.0
            )

            l2_snap = coin_to_l2.get(norm, L2OrderBookSnapshot(instrument_id=norm))

            asset_snapshots[norm] = AssetMarketSnapshot(
                instrument_id=norm,
                mid_price=mid_px,
                mark_price=mark_px,
                oracle_price=oracle_px,
                open_interest=oi,
                funding_rate_1h=funding_1h,
                funding_rate_8h_equiv=funding_8h_equiv,
                day_volume_usd=day_ntl_vol,
                l2_book=l2_snap,
                annualized_apr_pct=round(annualized_apr, 4),
                basis_bps=basis_bps,
                feed_source="HYPERLIQUID-L1",
                timestamp_ns=now_ns,
            )

        return MultiAssetMarketSnapshot(
            timestamp_ns=now_ns,
            assets=asset_snapshots,
            feed_source="HYPERLIQUID-L1",
        )

    def fetch_bybit_snapshot_sync(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        include_l2: bool = True,
    ) -> MultiAssetMarketSnapshot:
        """Synchronously poll Bybit V5 linear perps as failover."""
        now_ns = time.time_ns()

        def _fetch_single_bybit(coin_sym: str) -> tuple[str, AssetMarketSnapshot]:
            norm = normalize_coin_symbol(coin_sym)
            ticker_req = urllib.request.Request(
                f"{self.bybit_tickers_url}?category=linear&symbol={norm}USDT",
                headers={"User-Agent": "ProjectPhoenix/4.0"},
            )
            with urllib.request.urlopen(ticker_req, timeout=self.timeout_sec) as r_tick:
                t_data = json.loads(r_tick.read().decode("utf-8"))
            item = t_data["result"]["list"][0]

            last_px = float(item.get("lastPrice", 0.0))
            mark_px = float(item.get("markPrice", last_px))
            index_px = float(item.get("indexPrice", mark_px))
            mid_px = last_px or mark_px
            oi = float(item.get("openInterest", 0.0))
            turnover_24h = float(item.get("turnover24h", 0.0))
            funding_8h = float(item.get("fundingRate", 0.0))
            funding_1h = funding_8h / 8.0
            annualized_apr = funding_1h * 24.0 * 365.0 * 100.0
            basis_bps = (
                round((mark_px - index_px) / index_px * 10_000.0, 3) if index_px > 0 else 0.0
            )

            l2_snap = L2OrderBookSnapshot(instrument_id=norm)
            if include_l2:
                try:
                    book_req = urllib.request.Request(
                        f"{self.bybit_book_url}?category=linear&symbol={norm}USDT&limit=20",
                        headers={"User-Agent": "ProjectPhoenix/4.0"},
                    )
                    with urllib.request.urlopen(book_req, timeout=self.timeout_sec) as r_book:
                        b_data = json.loads(r_book.read().decode("utf-8"))
                    res = b_data.get("result", {})
                    raw_bids = res.get("b", [])[:20]
                    raw_asks = res.get("a", [])[:20]
                    time_ms = int(b_data.get("time", time.time() * 1000))
                    bids = [(float(b[0]), float(b[1])) for b in raw_bids]
                    asks = [(float(a[0]), float(a[1])) for a in raw_asks]
                    levels_b = [L2Level(price=b[0], size=b[1], order_count=1) for b in bids]
                    levels_a = [L2Level(price=a[0], size=a[1], order_count=1) for a in asks]
                    sp, d10, d50 = _compute_l2_depth_and_spread(bids, asks)
                    l2_snap = L2OrderBookSnapshot(
                        instrument_id=norm,
                        bids=bids,
                        asks=asks,
                        levels_bids=levels_b,
                        levels_asks=levels_a,
                        spread_bps=sp,
                        depth_10bps_usd=d10,
                        depth_50bps_usd=d50,
                        timestamp_ns=time_ms * 1_000_000,
                    )
                except Exception:
                    pass

            snap = AssetMarketSnapshot(
                instrument_id=norm,
                mid_price=mid_px,
                mark_price=mark_px,
                oracle_price=index_px,
                open_interest=oi,
                funding_rate_1h=funding_1h,
                funding_rate_8h_equiv=funding_8h,
                day_volume_usd=turnover_24h,
                l2_book=l2_snap,
                annualized_apr_pct=round(annualized_apr, 4),
                basis_bps=basis_bps,
                feed_source="BYBIT-V5-FALLBACK",
                timestamp_ns=now_ns,
            )
            return norm, snap

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(coins)) as executor:
            results = list(executor.map(_fetch_single_bybit, coins))

        return MultiAssetMarketSnapshot(
            timestamp_ns=now_ns,
            assets=dict(results),
            feed_source="BYBIT-V5-FALLBACK",
        )

    def get_multi_asset_snapshot_sync(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        include_l2: bool = True,
    ) -> MultiAssetMarketSnapshot:
        """Synchronously fetch multi-asset snapshot: HL L1 -> Bybit V5 -> Simulated fallback."""
        # 1. Try Hyperliquid L1 Primary
        try:
            snapshot = self.fetch_hyperliquid_snapshot_sync(coins=coins, include_l2=include_l2)
            self._cached_snapshot = snapshot
            self._last_success_source = "HYPERLIQUID-L1"
            self._last_success_ts_ns = snapshot.timestamp_ns
            return snapshot
        except Exception as hl_exc:
            logger.warning("Hyperliquid sync poll failed (%s), attempting Bybit V5 failover.", hl_exc)

        # 2. Try Bybit V5 Linear Fallback
        try:
            snapshot = self.fetch_bybit_snapshot_sync(coins=coins, include_l2=include_l2)
            self._cached_snapshot = snapshot
            self._last_success_source = "BYBIT-V5-FALLBACK"
            self._last_success_ts_ns = snapshot.timestamp_ns
            return snapshot
        except Exception as bybit_exc:
            logger.warning("Bybit sync failover failed (%s), activating simulated fallback.", bybit_exc)

        # 3. Simulated Fallback
        fallback = self.create_simulated_snapshot(coins=coins)
        self._cached_snapshot = fallback
        self._last_success_source = "SIMULATED-FALLBACK"
        self._last_success_ts_ns = fallback.timestamp_ns
        return fallback

    @property
    def latest_cached_snapshot(self) -> MultiAssetMarketSnapshot | None:
        """Return the most recently fetched multi-asset market snapshot."""
        return self._cached_snapshot

    @property
    def last_success_source(self) -> str:
        """Source of the latest successfully ingested market snapshot."""
        return self._last_success_source


# Global module singleton for application-wide ingestion
_GLOBAL_ENGINE: LiveMarketDataEngine | None = None


def get_market_data_engine() -> LiveMarketDataEngine:
    """Return or initialize the global LiveMarketDataEngine singleton."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = LiveMarketDataEngine()
    return _GLOBAL_ENGINE


def get_live_multi_asset_snapshot(
    coins: tuple[str, ...] = DEFAULT_ASSETS,
    include_l2: bool = True,
) -> MultiAssetMarketSnapshot:
    """Convenience synchronous function for retrieving the latest multi-asset snapshot."""
    return get_market_data_engine().get_multi_asset_snapshot_sync(
        coins=coins, include_l2=include_l2
    )


async def get_live_multi_asset_snapshot_async(
    coins: tuple[str, ...] = DEFAULT_ASSETS,
    include_l2: bool = True,
) -> MultiAssetMarketSnapshot:
    """Convenience asynchronous function for retrieving the latest multi-asset snapshot."""
    return await get_market_data_engine().get_multi_asset_snapshot(
        coins=coins, include_l2=include_l2
    )


# ==============================================================================
# Hyperliquid candle / funding history and bar feeds (REINFORCEMENT_PLAN P0.3 / P0.4)
# ==============================================================================


CANDLE_INTERVAL_MS: Final[dict[str, int]] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}
HORIZON_FOR_INTERVAL: Final[dict[str, Literal["15m", "1h", "4h"]]] = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
}
HYPERLIQUID_MAX_CANDLES: Final[int] = 5000


def _hl_info_post(payload: dict[str, Any], url: str = HYPERLIQUID_INFO_URL, timeout_sec: float = 2.5) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ProjectPhoenix/4.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_hyperliquid_candles(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
    url: str = HYPERLIQUID_INFO_URL,
    timeout_sec: float = 2.5,
) -> list[dict[str, Any]]:
    """Fetch raw Hyperliquid ``candleSnapshot`` rows (``t``, ``T``, ``o``, ``h``, ``l``, ``c``, ``v``, ``n``).

    Returns at most the venue's page limit (5000 candles), oldest first.
    """
    if interval not in CANDLE_INTERVAL_MS:
        raise ValueError(f"Unsupported candle interval '{interval}'")
    resolved_end = int(end_ms) if end_ms is not None else int(time.time() * 1000)
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": normalize_coin_symbol(coin),
            "interval": interval,
            "startTime": int(start_ms),
            "endTime": resolved_end,
        },
    }
    data = _hl_info_post(payload, url=url, timeout_sec=timeout_sec)
    if not isinstance(data, list):
        raise ValueError("Unexpected candleSnapshot payload shape")
    rows = [row for row in data if isinstance(row, dict) and "t" in row and "c" in row]
    rows.sort(key=lambda r: int(r["t"]))
    return rows


def fetch_hyperliquid_funding_history(
    coin: str,
    start_ms: int,
    end_ms: int | None = None,
    url: str = HYPERLIQUID_INFO_URL,
    timeout_sec: float = 2.5,
) -> list[tuple[int, float]]:
    """Fetch hourly Hyperliquid funding settlements as ``(time_ms, funding_rate_1h)`` oldest first."""
    payload: dict[str, Any] = {
        "type": "fundingHistory",
        "coin": normalize_coin_symbol(coin),
        "startTime": int(start_ms),
    }
    if end_ms is not None:
        payload["endTime"] = int(end_ms)
    data = _hl_info_post(payload, url=url, timeout_sec=timeout_sec)
    if not isinstance(data, list):
        raise ValueError("Unexpected fundingHistory payload shape")
    out: list[tuple[int, float]] = []
    for row in data:
        if isinstance(row, dict) and "time" in row and "fundingRate" in row:
            out.append((int(row["time"]), float(row["fundingRate"])))
    out.sort(key=lambda r: r[0])
    return out


def funding_rate_for_time(
    funding_history: Sequence[tuple[int, float]],
    ts_ms: int,
    default: float = 0.0,
) -> float:
    """Most recent funding settlement at or before ``ts_ms`` (strictly causal lookup)."""
    best = default
    for t_ms, rate in funding_history:
        if t_ms <= ts_ms:
            best = rate
        else:
            break
    return best


def candle_to_dollar_bar(
    candle: Mapping[str, Any],
    *,
    instrument_id: str,
    horizon_label: Literal["15m", "1h", "4h"],
    open_interest: float,
    funding_rate_1h: float,
    mark_price: float | None = None,
    spot_price: float | None = None,
) -> DollarBar:
    """Convert one Hyperliquid candle into a ``DollarBar`` without synthesising unavailable fields.

    Taker buy/sell split is not part of the candle, so dollar volume is split evenly and the
    feed reports ``taker_flow=False``. Liquidation volume is ``0.0`` and reported unavailable.
    """
    t_ms = int(candle["t"])
    close_ms = int(candle.get("T", t_ms + CANDLE_INTERVAL_MS["15m"] - 1)) + 1
    o_px = float(candle["o"])
    h_px = float(candle["h"])
    l_px = float(candle["l"])
    c_px = float(candle["c"])
    vol = max(0.0, float(candle.get("v", 0.0)))
    n_trades = max(1, int(candle.get("n", 1)))
    dvol = vol * c_px
    return DollarBar(
        instrument_id=instrument_id,
        horizon_label=horizon_label,
        ts_open_ns=t_ms * 1_000_000,
        ts_close_ns=close_ms * 1_000_000,
        open=o_px,
        high=max(h_px, o_px, c_px),
        low=min(l_px, o_px, c_px),
        close=c_px,
        volume=vol,
        dollar_volume=dvol,
        buy_dollar_volume=dvol * 0.5,
        sell_dollar_volume=dvol * 0.5,
        open_interest=max(0.0, float(open_interest)),
        funding_rate_1h=float(funding_rate_1h),
        funding_rate_8h=float(funding_rate_1h) * 8.0,
        mark_price=float(mark_price) if mark_price is not None and mark_price > 0.0 else c_px,
        spot_price=float(spot_price) if spot_price is not None and spot_price > 0.0 else c_px,
        liquidation_volume_usd=0.0,
        tick_count=n_trades,
    )


def perp_instrument_id(coin: str) -> str:
    """Instrument id used for persisted bars (legacy ``BTCUSDT-PERP`` naming kept for the Parquet schema)."""
    return f"{normalize_coin_symbol(coin)}USDT-PERP"


class BarFeed(Protocol):
    """Source of closed bars and market snapshots for the paper-trading loop.

    The live implementation reads Hyperliquid candles; replay and tests supply scripted bars
    through the same interface so the tick code path is identical.
    """

    def snapshot(self) -> MultiAssetMarketSnapshot:
        """Current multi-asset market snapshot."""
        ...

    def seed_bars(self, coin: str, n_bars: int, interval: str = "15m") -> tuple[list[DollarBar], DataAvailability]:
        """Historical closed bars for ``coin`` (oldest first) and what was really available."""
        ...

    def hourly_closes(self, coin: str, n_hours: int) -> list[tuple[int, float]]:
        """Hourly ``(ts_ms, close)`` history for HRP return matrices."""
        ...

    def next_closed_bars(
        self, after_ts_open_ms: int, snapshot: MultiAssetMarketSnapshot
    ) -> dict[str, DollarBar] | None:
        """Bars for the next closed 15m candle after ``after_ts_open_ms``, keyed by coin, or None."""
        ...

    @property
    def availability(self) -> DataAvailability:
        """Availability flags of the most recent poll."""
        ...


class HyperliquidCandleBarFeed:
    """Live bar feed backed by Hyperliquid ``candleSnapshot`` and ``fundingHistory``."""

    def __init__(
        self,
        coins: tuple[str, ...] = DEFAULT_ASSETS,
        engine: LiveMarketDataEngine | None = None,
        url: str = HYPERLIQUID_INFO_URL,
        timeout_sec: float = 2.5,
        close_grace_ms: int = 2_000,
    ) -> None:
        self._coins: tuple[str, ...] = tuple(normalize_coin_symbol(c) for c in coins)
        self._engine = engine or get_market_data_engine()
        self._url = url
        self._timeout = timeout_sec
        self._close_grace_ms = close_grace_ms
        self._availability = DataAvailability(feed_source="HYPERLIQUID-L1")

    @property
    def coins(self) -> tuple[str, ...]:
        """Coins polled on every bar."""
        return self._coins

    @property
    def availability(self) -> DataAvailability:
        """Availability flags of the most recent poll."""
        return self._availability

    def snapshot(self) -> MultiAssetMarketSnapshot:
        """Current live snapshot (Hyperliquid, Bybit failover, simulated fallback flagged by source)."""
        return self._engine.get_multi_asset_snapshot_sync(coins=self._coins, include_l2=True)

    def seed_bars(self, coin: str, n_bars: int, interval: str = "15m") -> tuple[list[DollarBar], DataAvailability]:
        """Fetch up to ``n_bars`` closed candles with real hourly funding; OI history is not available."""
        norm = normalize_coin_symbol(coin)
        step_ms = CANDLE_INTERVAL_MS[interval]
        now_ms = int(time.time() * 1000)
        n_req = max(1, min(int(n_bars), HYPERLIQUID_MAX_CANDLES))
        start_ms = now_ms - (n_req + 2) * step_ms
        candles = fetch_hyperliquid_candles(norm, interval, start_ms, now_ms, url=self._url, timeout_sec=self._timeout)
        closed = [c for c in candles if int(c.get("T", 0)) + 1 <= now_ms - self._close_grace_ms]
        closed = closed[-n_req:]
        if not closed:
            avail = DataAvailability(feed_source="HYPERLIQUID-L1")
            self._availability = avail
            return [], avail

        funding_hist: list[tuple[int, float]] = []
        funding_ok = False
        try:
            funding_hist = fetch_hyperliquid_funding_history(
                norm, int(closed[0]["t"]) - 3_600_000, now_ms, url=self._url, timeout_sec=self._timeout
            )
            funding_ok = len(funding_hist) > 0
        except Exception as exc:
            logger.warning("Hyperliquid fundingHistory unavailable for %s (%s); funding zero-filled.", norm, exc)

        current_oi = 0.0
        try:
            snap = self._engine.latest_cached_snapshot or self.snapshot()
            asset = snap.get_asset(norm)
            if asset is not None and snap.feed_source != "SIMULATED-FALLBACK":
                current_oi = float(asset.open_interest)
        except Exception:
            current_oi = 0.0

        horizon = HORIZON_FOR_INTERVAL.get(interval, "15m")
        inst = perp_instrument_id(norm)
        bars = [
            candle_to_dollar_bar(
                c,
                instrument_id=inst,
                horizon_label=horizon,
                open_interest=current_oi,
                funding_rate_1h=funding_rate_for_time(funding_hist, int(c["t"]), default=0.0),
            )
            for c in closed
        ]
        avail = DataAvailability(
            candles=True,
            taker_flow=False,
            liquidations=False,
            open_interest_history=False,
            funding_history=funding_ok,
            feed_source="HYPERLIQUID-L1",
        )
        self._availability = avail
        return bars, avail

    def hourly_closes(self, coin: str, n_hours: int) -> list[tuple[int, float]]:
        """Closed 1h candles as ``(ts_ms, close)``."""
        norm = normalize_coin_symbol(coin)
        now_ms = int(time.time() * 1000)
        n_req = max(2, min(int(n_hours), HYPERLIQUID_MAX_CANDLES))
        start_ms = now_ms - (n_req + 2) * CANDLE_INTERVAL_MS["1h"]
        candles = fetch_hyperliquid_candles(norm, "1h", start_ms, now_ms, url=self._url, timeout_sec=self._timeout)
        closed = [c for c in candles if int(c.get("T", 0)) + 1 <= now_ms - self._close_grace_ms]
        return [(int(c["t"]), float(c["c"])) for c in closed[-n_req:]]

    def next_closed_bars(
        self, after_ts_open_ms: int, snapshot: MultiAssetMarketSnapshot
    ) -> dict[str, DollarBar] | None:
        """Return the next closed 15m candle after ``after_ts_open_ms`` for every coin, or None.

        The first coin in ``coins`` sets the bar clock. Open interest, funding, mark and oracle
        prices come from the live snapshot at poll time. A simulated snapshot yields no bars.
        """
        if snapshot.feed_source == "SIMULATED-FALLBACK":
            self._availability = DataAvailability(feed_source="SIMULATED-FALLBACK")
            return None
        now_ms = int(time.time() * 1000)
        step_ms = CANDLE_INTERVAL_MS["15m"]
        start_ms = int(after_ts_open_ms) + 1
        primary = self._coins[0]
        try:
            primary_candles = fetch_hyperliquid_candles(
                primary, "15m", start_ms, now_ms, url=self._url, timeout_sec=self._timeout
            )
        except Exception as exc:
            logger.warning("Hyperliquid candleSnapshot poll failed (%s); no bar this tick.", exc)
            self._availability = self._availability.model_copy(update={"candles": False})
            return None
        closed = [
            c
            for c in primary_candles
            if int(c["t"]) > int(after_ts_open_ms) and int(c.get("T", int(c["t"]) + step_ms - 1)) + 1 <= now_ms - self._close_grace_ms
        ]
        if not closed:
            return None
        target = closed[0]
        t_ms = int(target["t"])

        bars: dict[str, DollarBar] = {}
        for coin in self._coins:
            asset = snapshot.get_asset(coin)
            oi = float(asset.open_interest) if asset is not None else 0.0
            f1h = float(asset.funding_rate_1h) if asset is not None else 0.0
            mark = float(asset.mark_price) if asset is not None else None
            oracle = float(asset.oracle_price) if asset is not None else None
            if coin == primary:
                candle: Mapping[str, Any] | None = target
            else:
                try:
                    rows = fetch_hyperliquid_candles(
                        coin, "15m", t_ms, t_ms + step_ms - 1, url=self._url, timeout_sec=self._timeout
                    )
                except Exception as exc:
                    logger.warning("Hyperliquid candle for %s at %d unavailable (%s).", coin, t_ms, exc)
                    rows = []
                candle = next((r for r in rows if int(r["t"]) == t_ms), None)
            if candle is None:
                continue
            bars[coin] = candle_to_dollar_bar(
                candle,
                instrument_id=perp_instrument_id(coin),
                horizon_label="15m",
                open_interest=oi,
                funding_rate_1h=f1h,
                mark_price=mark,
                spot_price=oracle,
            )
        self._availability = DataAvailability(
            candles=True,
            taker_flow=False,
            liquidations=False,
            open_interest_history=False,
            funding_history=True,
            feed_source=str(snapshot.feed_source),
        )
        return bars or None


class ScriptedBarFeed:
    """Deterministic bar feed for replay and tests: pre-built bars and snapshots, no network.

    ``bars_by_coin`` holds chronologically ordered 15m bars per coin; ``history_by_coin`` is
    returned by ``seed_bars``; ``snapshots`` are returned in order by ``snapshot()`` (the last one
    repeats once exhausted).
    """

    def __init__(
        self,
        bars_by_coin: Mapping[str, Sequence[DollarBar]],
        snapshots: Sequence[MultiAssetMarketSnapshot],
        history_by_coin: Mapping[str, Sequence[DollarBar]] | None = None,
        hourly_closes_by_coin: Mapping[str, Sequence[tuple[int, float]]] | None = None,
        availability: DataAvailability | None = None,
    ) -> None:
        if not snapshots:
            raise ValueError("ScriptedBarFeed requires at least one snapshot")
        self._bars: dict[str, list[DollarBar]] = {
            normalize_coin_symbol(k): list(v) for k, v in bars_by_coin.items()
        }
        self._history: dict[str, list[DollarBar]] = {
            normalize_coin_symbol(k): list(v) for k, v in (history_by_coin or {}).items()
        }
        self._hourly: dict[str, list[tuple[int, float]]] = {
            normalize_coin_symbol(k): [(int(t), float(p)) for t, p in v]
            for k, v in (hourly_closes_by_coin or {}).items()
        }
        self._snapshots: list[MultiAssetMarketSnapshot] = list(snapshots)
        self._snapshot_cursor: int = 0
        self._availability = availability or DataAvailability(
            candles=True, funding_history=True, feed_source="SCRIPTED"
        )

    @property
    def availability(self) -> DataAvailability:
        """Availability flags declared for the scripted data."""
        return self._availability

    @property
    def remaining_bars(self) -> int:
        """Bars left for the primary coin (the first key)."""
        if not self._bars:
            return 0
        first = next(iter(self._bars))
        return len(self._bars[first])

    def snapshot(self) -> MultiAssetMarketSnapshot:
        """Next scripted snapshot; the last snapshot repeats once the script is exhausted."""
        idx = min(self._snapshot_cursor, len(self._snapshots) - 1)
        self._snapshot_cursor += 1
        return self._snapshots[idx]

    def seed_bars(self, coin: str, n_bars: int, interval: str = "15m") -> tuple[list[DollarBar], DataAvailability]:
        """Scripted history for ``coin``."""
        hist = self._history.get(normalize_coin_symbol(coin), [])
        return list(hist[-max(0, int(n_bars)) :]) if n_bars > 0 else [], self._availability

    def hourly_closes(self, coin: str, n_hours: int) -> list[tuple[int, float]]:
        """Scripted hourly closes for ``coin``."""
        series = self._hourly.get(normalize_coin_symbol(coin), [])
        return list(series[-max(0, int(n_hours)) :])

    def next_closed_bars(
        self, after_ts_open_ms: int, snapshot: MultiAssetMarketSnapshot
    ) -> dict[str, DollarBar] | None:
        """Pop the next bar per coin whose open is after ``after_ts_open_ms``."""
        out: dict[str, DollarBar] = {}
        for coin, bars in self._bars.items():
            while bars and bars[0].ts_open_ns // 1_000_000 <= int(after_ts_open_ms):
                bars.pop(0)
            if bars:
                out[coin] = bars.pop(0)
        return out or None
