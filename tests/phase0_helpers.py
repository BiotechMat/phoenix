"""Deterministic, network-free fixtures for the Phase 0 live-loop integrity tests.

Nothing here is random: histories are flat or sinusoidal, price paths are scripted bar by bar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

from core.contracts import (
    AssetMarketSnapshot,
    DataAvailability,
    DeribitOptionsSurface,
    DollarBar,
    HyperliquidWhaleVaultMetrics,
    MultiAssetMarketSnapshot,
)
from core.data_ingestion import ScriptedBarFeed
from sleeves.sleeve_1_derivatives_alpha import DerivativesAlphaConfig, DerivativesAlphaSignal, DerivativesAlphaSleeve
from sleeves.sleeve_3_funding_carry import FundingCarrySleeve
from ui.api_server import ControlCenterStateStore

BASE_TS_MS = 1_760_000_000_000
STEP_MS = 900_000
ASSETS = ("BTC", "ETH", "SOL")
DEFAULT_PRICES: dict[str, float] = {"BTC": 100.0, "ETH": 10.0, "SOL": 5.0}


def make_bar(
    coin: str,
    index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    funding_1h: float = 0.0,
    open_interest: float = 1000.0,
    volume: float = 10.0,
) -> DollarBar:
    """One 15m bar at ``BASE_TS_MS + index * STEP_MS`` (negative indices are history)."""
    t_open_ns = (BASE_TS_MS + index * STEP_MS) * 1_000_000
    hi = max(high, open_, close)
    lo = min(low, open_, close)
    dvol = volume * close
    return DollarBar(
        instrument_id=f"{coin}USDT-PERP",
        horizon_label="15m",
        ts_open_ns=t_open_ns,
        ts_close_ns=t_open_ns + STEP_MS * 1_000_000,
        open=open_,
        high=hi,
        low=lo,
        close=close,
        volume=volume,
        dollar_volume=dvol,
        buy_dollar_volume=dvol * 0.5,
        sell_dollar_volume=dvol * 0.5,
        open_interest=open_interest,
        funding_rate_1h=funding_1h,
        funding_rate_8h=funding_1h * 8.0,
        mark_price=close,
        spot_price=close,
        liquidation_volume_usd=0.0,
        tick_count=100,
    )


def flat_history(coin: str, n_bars: int, price: float, *, funding_1h: float = 0.0) -> list[DollarBar]:
    """``n_bars`` identical bars ending just before index 0."""
    return [
        make_bar(coin, -n_bars + i, price, price, price, price, funding_1h=funding_1h)
        for i in range(n_bars)
    ]


def flat_bars(coin: str, start_index: int, n_bars: int, price: float, *, funding_1h: float = 0.0) -> list[DollarBar]:
    """``n_bars`` flat live bars starting at ``start_index``."""
    return [
        make_bar(coin, start_index + i, price, price, price, price, funding_1h=funding_1h)
        for i in range(n_bars)
    ]


def sine_history(
    coin: str,
    n_bars: int,
    price: float,
    *,
    amplitude_pct: float = 0.03,
    period_bars: int = 64,
    funding_1h: float = 0.00001,
) -> list[DollarBar]:
    """Deterministic oscillating history with drifting open interest (yields two-class labels)."""
    bars: list[DollarBar] = []
    prev_close = price
    for i in range(n_bars):
        phase = 2.0 * math.pi * i / period_bars
        close = price * (1.0 + amplitude_pct * math.sin(phase))
        oi = 1000.0 * (1.0 + 0.0005 * i) + 40.0 * math.sin(phase + 1.0)
        hi = max(prev_close, close) * 1.0005
        lo = min(prev_close, close) * 0.9995
        bars.append(
            make_bar(coin, -n_bars + i, prev_close, hi, lo, close, funding_1h=funding_1h, open_interest=oi)
        )
        prev_close = close
    return bars


def ohlc_path(
    coin: str,
    steps: Sequence[tuple[float, float, float, float]],
    *,
    start_index: int = 0,
    funding_1h: float = 0.0,
) -> list[DollarBar]:
    """Scripted live bars from ``(open, high, low, close)`` tuples."""
    return [
        make_bar(coin, start_index + i, o, h, lo, c, funding_1h=funding_1h)
        for i, (o, h, lo, c) in enumerate(steps)
    ]


def make_snapshot(
    prices: Mapping[str, float] | None = None,
    funding_1h: Mapping[str, float] | None = None,
) -> MultiAssetMarketSnapshot:
    """Neutral multi-asset snapshot (zero funding unless given, mark = oracle = mid)."""
    px = dict(DEFAULT_PRICES if prices is None else prices)
    fund = dict(funding_1h or {})
    ts_ns = BASE_TS_MS * 1_000_000
    assets = {
        coin: AssetMarketSnapshot(
            instrument_id=coin,
            mid_price=p,
            mark_price=p,
            oracle_price=p,
            open_interest=1000.0,
            funding_rate_1h=fund.get(coin, 0.0),
            day_volume_usd=1_000_000.0,
            timestamp_ns=ts_ns,
        )
        for coin, p in px.items()
    }
    return MultiAssetMarketSnapshot(timestamp_ns=ts_ns, assets=assets, feed_source="HYPERLIQUID-L1")


def hourly_closes_trending(coin: str, n_hours: int, price: float, slope_bps: float, wobble_bps: float) -> list[tuple[int, float]]:
    """Deterministic hourly closes: linear drift plus a cosine wobble."""
    out: list[tuple[int, float]] = []
    for h in range(n_hours):
        p = price * (1.0 + slope_bps * 1e-4 * h + wobble_bps * 1e-4 * math.cos(h / 2.0))
        out.append(((BASE_TS_MS - (n_hours - h) * 3_600_000), p))
    return out


class ForcedSignalDerivativesSleeve(DerivativesAlphaSleeve):
    """Derivatives sleeve stub that emits a fixed directional signal for one asset and never fetches."""

    def __init__(self, asset: str = "BTC", direction: int = 1, p_meta: float = 0.80, edge_bps: float = 60.0) -> None:
        # The forced stub stands in for a validated sleeve, so it is allowed to allocate.
        super().__init__(DerivativesAlphaConfig(allocate=True))
        self._asset = asset
        self._direction = direction
        self._p = p_meta
        self._edge = edge_bps

    def fetch_deribit_options_surface_sync(self, currency: str = "BTC") -> DeribitOptionsSurface:
        return DeribitOptionsSurface(currency=currency)

    def fetch_hyperliquid_vault_positioning_sync(self, vault_address: str | None = None) -> HyperliquidWhaleVaultMetrics:
        return HyperliquidWhaleVaultMetrics()

    def generate_alpha_signal(
        self,
        options_surface: DeribitOptionsSurface | None = None,
        vault_metrics: HyperliquidWhaleVaultMetrics | None = None,
        bar: DollarBar | None = None,
        instrument_id: str = "BTC",
        timestamp_ns: int | None = None,
    ) -> DerivativesAlphaSignal:
        clean = instrument_id.strip().upper().removesuffix("USDT-PERP").removesuffix("-PERP")
        active = clean == self._asset and self._direction != 0
        direction = self._direction if active else 0
        return DerivativesAlphaSignal(
            instrument_id=clean,
            primary_direction=1 if direction > 0 else (-1 if direction < 0 else 0),
            p_meta=self._p if active else 0.5,
            expected_gross_edge_bps=self._edge if active else 0.0,
            skew_25d=0.02 if active else 0.0,
            term_structure_slope=0.0,
            gex_regime="NEUTRAL",
            hlp_vault_crowding="BALANCED",
            feature_attribution={"forced": 1.0 if active else 0.0},
            timestamp_ns=timestamp_ns or 0,
        )


def build_store(
    tmp_path: Path,
    *,
    live_bars: Mapping[str, Sequence[DollarBar]],
    snapshots: Sequence[MultiAssetMarketSnapshot],
    history: Mapping[str, Sequence[DollarBar]] | None = None,
    hourly_closes: Mapping[str, Sequence[tuple[int, float]]] | None = None,
    availability: DataAvailability | None = None,
    derivatives_sleeve: DerivativesAlphaSleeve | None = None,
    carry_sleeve: FundingCarrySleeve | None = None,
    ledger_name: str = "ledger.json",
    parquet_name: str = "bars.parquet",
) -> ControlCenterStateStore:
    """Construct a store on a scripted feed with isolated ledger and Parquet paths (no network)."""
    feed = ScriptedBarFeed(
        bars_by_coin=live_bars,
        snapshots=snapshots,
        history_by_coin=history,
        hourly_closes_by_coin=hourly_closes,
        availability=availability,
    )
    return ControlCenterStateStore(
        ledger_path=tmp_path / ledger_name,
        parquet_path=tmp_path / parquet_name,
        feed=feed,
        derivatives_sleeve=derivatives_sleeve or ForcedSignalDerivativesSleeve(direction=0),
        carry_sleeve=carry_sleeve,
        fetch_derivatives_inputs=False,
    )


def barrier_path_steps() -> list[tuple[float, float, float, float]]:
    """BTC price path that, with a standing long signal, closes exactly three trades.

    Flat history makes ``sigma_bar`` hit the 1e-4 floor, so the barriers are the fee-floor
    levels: +80 bps profit, -20 bps stop, 16-bar vertical.

    - bar 0: entry at 100.00
    - bar 1: high 101.00 touches the +80 bps barrier (100.80): PROFIT; re-entry at 100.50
    - bar 2: low 100.00 touches the -20 bps barrier (100.299): STOP; re-entry at 100.20
    - bars 3..18: quiet inside both barriers; bar 18 is the 16th bar after entry: VERTICAL
    """
    steps: list[tuple[float, float, float, float]] = [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 101.0, 99.95, 100.5),
        (100.5, 100.6, 100.0, 100.2),
    ]
    steps.extend([(100.2, 100.25, 100.15, 100.2)] * 16)
    return steps


def companion_flat_bars(n_bars: int, start_index: int = 0) -> dict[str, list[DollarBar]]:
    """Flat ETH and SOL bars so every asset has marks on every scripted bar."""
    return {
        "ETH": flat_bars("ETH", start_index, n_bars, DEFAULT_PRICES["ETH"]),
        "SOL": flat_bars("SOL", start_index, n_bars, DEFAULT_PRICES["SOL"]),
    }


def ledger_core_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    """Fields of a persisted ledger that must not change in a zero-signal replay."""
    keys = (
        "base_capital_usd",
        "current_equity_usd",
        "total_realized_pnl",
        "total_fees_paid_usd",
        "total_funding_pnl_usd",
        "total_trades",
        "win_count",
        "loss_count",
    )
    return {k: data.get(k) for k in keys} | {"trade_history_len": len(data.get("trade_history", []))}
