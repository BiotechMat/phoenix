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
from typing import Any, Final, Literal
import urllib.error
import urllib.request

import httpx

from core.contracts import (
    AssetMarketSnapshot,
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
