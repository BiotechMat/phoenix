"""Native NautilusTrader DataActor for Dynamic Dollar Bars and Derivatives Positioning Alpha.

Aggregates exchange trade ticks, funding rates, mark/spot prices, Open Interest, and
liquidation streams into Dynamic Dollar Bars (~15m, 1h, and 4h horizons) using Numba and
Polars, then emits strictly causal `PositioningSignalSnapshot` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numba
import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import DollarBar, FFDWeightsConfig, PositioningSignalSnapshot
from core.features.positioning import PositioningFeatureEngine

try:
    from nautilus_trader.common.actor import DataActor as _NautilusDataActor
    from nautilus_trader.common.actor import DataActorConfig as _NautilusDataActorConfig

    _HAS_NAUTILUS_ACTOR = True
except ImportError:
    _HAS_NAUTILUS_ACTOR = False

    class _NautilusDataActorConfig:  # type: ignore[no-redef]
        """Typed fallback DataActorConfig when NautilusTrader Cython binaries are absent."""

        def __init__(self, actor_id: str | None = None, **kwargs: Any) -> None:
            self.actor_id = actor_id or "DerivativesFeatureActor-001"
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _NautilusDataActor:  # type: ignore[no-redef]
        """Typed fallback DataActor base class for unit-test environments."""

        def __init__(self, config: Any | None = None) -> None:
            self.config = config
            self._is_running: bool = False

        def on_start(self) -> None:
            self._is_running = True

        def on_stop(self) -> None:
            self._is_running = False

        def on_reset(self) -> None:
            pass


HORIZON_BARS_PER_DAY: dict[Literal["15m", "1h", "4h"], int] = {
    "15m": 96,
    "1h": 24,
    "4h": 6,
}


class DerivativesActorConfig(BaseModel):
    """Configuration for DerivativesFeatureActor and Dynamic Dollar Bar sampling."""

    model_config = ConfigDict(frozen=True)

    actor_id: str = "DerivativesFeatureActor-001"
    instrument_id: str = "BTCUSDT-PERP.BINANCE"
    horizon_label: Literal["15m", "1h", "4h"] = "15m"
    target_dollar_volume: float = Field(
        default=25_000_000.0,
        gt=0.0,
        description="Initial dollar volume threshold per bar",
    )
    adaptive_threshold: bool = Field(
        default=True,
        description="Whether to adapt dollar threshold toward trailing daily volume / target_bars_per_day",
    )
    default_ffd_d: float = Field(default=0.35, ge=0.05, le=0.60)
    zscore_window: int = Field(default=48, ge=5)
    max_bar_history: int = Field(default=2048, ge=32)


@numba.njit(cache=True)
def _aggregate_ticks_numba(
    ts_ns: npt.NDArray[np.int64],
    prices: npt.NDArray[np.float64],
    sizes: npt.NDArray[np.float64],
    is_buy: npt.NDArray[np.uint8],
    open_interest: npt.NDArray[np.float64],
    funding_rate_8h: npt.NDArray[np.float64],
    mark_prices: npt.NDArray[np.float64],
    spot_prices: npt.NDArray[np.float64],
    liq_usd: npt.NDArray[np.float64],
    initial_threshold: float,
    adaptive: bool,
    bars_per_day: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Numba JIT kernel to aggregate raw tick arrays into Dynamic Dollar Bars in a single pass."""
    n = prices.shape[0]
    out_ts_open = np.empty(n, dtype=np.int64)
    out_ts_close = np.empty(n, dtype=np.int64)
    out_open = np.empty(n, dtype=np.float64)
    out_high = np.empty(n, dtype=np.float64)
    out_low = np.empty(n, dtype=np.float64)
    out_close = np.empty(n, dtype=np.float64)
    out_vol = np.empty(n, dtype=np.float64)
    out_dvol = np.empty(n, dtype=np.float64)
    out_buy_dvol = np.empty(n, dtype=np.float64)
    out_sell_dvol = np.empty(n, dtype=np.float64)
    out_oi = np.empty(n, dtype=np.float64)
    out_funding = np.empty(n, dtype=np.float64)
    out_mark = np.empty(n, dtype=np.float64)
    out_spot = np.empty(n, dtype=np.float64)
    out_liq = np.empty(n, dtype=np.float64)
    out_ticks = np.empty(n, dtype=np.int64)

    bar_count = 0
    if n == 0:
        return (
            out_ts_open[:0],
            out_ts_close[:0],
            out_open[:0],
            out_high[:0],
            out_low[:0],
            out_close[:0],
            out_vol[:0],
            out_dvol[:0],
            out_buy_dvol[:0],
            out_sell_dvol[:0],
            out_oi[:0],
            out_funding[:0],
            out_mark[:0],
            out_spot[:0],
            out_liq[:0],
            out_ticks[:0],
        )

    threshold = initial_threshold
    target_ns_per_bar = (86_400.0 * 1e9) / float(max(bars_per_day, 1))

    in_bar = False
    cur_ts_open: np.int64 = np.int64(0)
    cur_open = 0.0
    cur_high = 0.0
    cur_low = 0.0
    cur_vol = 0.0
    cur_dvol = 0.0
    cur_buy_dvol = 0.0
    cur_sell_dvol = 0.0
    cur_liq = 0.0
    cur_ticks: np.int64 = np.int64(0)

    for i in range(n):
        p = prices[i]
        s = sizes[i]
        dv = p * s
        if not in_bar:
            in_bar = True
            cur_ts_open = ts_ns[i]
            cur_open = p
            cur_high = p
            cur_low = p
            cur_vol = 0.0
            cur_dvol = 0.0
            cur_buy_dvol = 0.0
            cur_sell_dvol = 0.0
            cur_liq = 0.0
            cur_ticks = np.int64(0)

        if p > cur_high:
            cur_high = p
        if p < cur_low:
            cur_low = p
        cur_vol += s
        cur_dvol += dv
        if is_buy[i] != 0:
            cur_buy_dvol += dv
        else:
            cur_sell_dvol += dv
        cur_liq += liq_usd[i]
        cur_ticks += np.int64(1)

        if cur_dvol >= threshold:
            out_ts_open[bar_count] = cur_ts_open
            out_ts_close[bar_count] = ts_ns[i]
            out_open[bar_count] = cur_open
            out_high[bar_count] = cur_high
            out_low[bar_count] = cur_low
            out_close[bar_count] = p
            out_vol[bar_count] = cur_vol
            out_dvol[bar_count] = cur_dvol
            out_buy_dvol[bar_count] = cur_buy_dvol
            out_sell_dvol[bar_count] = cur_sell_dvol
            out_oi[bar_count] = open_interest[i]
            out_funding[bar_count] = funding_rate_8h[i]
            out_mark[bar_count] = mark_prices[i]
            out_spot[bar_count] = spot_prices[i]
            out_liq[bar_count] = cur_liq
            out_ticks[bar_count] = cur_ticks
            bar_count += 1

            if adaptive:
                elapsed_ns = float(max(ts_ns[i] - cur_ts_open, np.int64(1_000_000_000)))
                implied_target = cur_dvol * (target_ns_per_bar / elapsed_ns)
                clipped_target = min(max(implied_target, initial_threshold * 0.25), initial_threshold * 4.0)
                threshold = 0.90 * threshold + 0.10 * clipped_target

            in_bar = False

    return (
        out_ts_open[:bar_count].copy(),
        out_ts_close[:bar_count].copy(),
        out_open[:bar_count].copy(),
        out_high[:bar_count].copy(),
        out_low[:bar_count].copy(),
        out_close[:bar_count].copy(),
        out_vol[:bar_count].copy(),
        out_dvol[:bar_count].copy(),
        out_buy_dvol[:bar_count].copy(),
        out_sell_dvol[:bar_count].copy(),
        out_oi[:bar_count].copy(),
        out_funding[:bar_count].copy(),
        out_mark[:bar_count].copy(),
        out_spot[:bar_count].copy(),
        out_liq[:bar_count].copy(),
        out_ticks[:bar_count].copy(),
    )


def aggregate_dollar_bars_polars(
    ticks_df: pl.DataFrame,
    instrument_id: str = "BTCUSDT-PERP.BINANCE",
    horizon_label: Literal["15m", "1h", "4h"] = "15m",
    target_dollar_volume: float = 25_000_000.0,
    adaptive_threshold: bool = True,
) -> list[DollarBar]:
    """Aggregate a Polars DataFrame of exchange ticks into Dynamic Dollar Bars via Numba.

    Expected columns in `ticks_df`:
    - `ts_ns` (int64)
    - `price` (float64)
    - `size` (float64)
    Optional columns (filled with sensible defaults if absent):
    - `is_buy` (bool or uint8, default True)
    - `open_interest` (float64, default 100_000.0)
    - `funding_rate_8h` (float64, default 0.0001)
    - `mark_price` (float64, default `price`)
    - `spot_price` (float64, default `price`)
    - `liquidation_volume_usd` (float64, default 0.0)
    """
    if ticks_df.height == 0:
        return []

    n = ticks_df.height
    cols = set(ticks_df.columns)
    for req in ("ts_ns", "price", "size"):
        if req not in cols:
            raise ValueError(f"ticks_df missing required column '{req}'")

    ts_ns = ticks_df.get_column("ts_ns").cast(pl.Int64).to_numpy()
    prices = ticks_df.get_column("price").cast(pl.Float64).to_numpy()
    sizes = ticks_df.get_column("size").cast(pl.Float64).to_numpy()

    if "is_buy" in cols:
        is_buy = ticks_df.get_column("is_buy").cast(pl.UInt8).to_numpy()
    else:
        is_buy = np.ones(n, dtype=np.uint8)

    if "open_interest" in cols:
        oi = ticks_df.get_column("open_interest").cast(pl.Float64).to_numpy()
    else:
        oi = np.full(n, 100_000.0, dtype=np.float64)

    if "funding_rate_8h" in cols:
        funding = ticks_df.get_column("funding_rate_8h").cast(pl.Float64).to_numpy()
    else:
        funding = np.full(n, 0.0001, dtype=np.float64)

    if "mark_price" in cols:
        mark = ticks_df.get_column("mark_price").cast(pl.Float64).to_numpy()
    else:
        mark = prices.copy()

    if "spot_price" in cols:
        spot = ticks_df.get_column("spot_price").cast(pl.Float64).to_numpy()
    else:
        spot = prices.copy()

    if "liquidation_volume_usd" in cols:
        liq = ticks_df.get_column("liquidation_volume_usd").cast(pl.Float64).to_numpy()
    else:
        liq = np.zeros(n, dtype=np.float64)

    bars_per_day = HORIZON_BARS_PER_DAY[horizon_label]
    res = _aggregate_ticks_numba(
        ts_ns=ts_ns,
        prices=prices,
        sizes=sizes,
        is_buy=is_buy,
        open_interest=oi,
        funding_rate_8h=funding,
        mark_prices=mark,
        spot_prices=spot,
        liq_usd=liq,
        initial_threshold=float(target_dollar_volume),
        adaptive=adaptive_threshold,
        bars_per_day=bars_per_day,
    )

    (
        b_ts_open,
        b_ts_close,
        b_open,
        b_high,
        b_low,
        b_close,
        b_vol,
        b_dvol,
        b_buy_dvol,
        b_sell_dvol,
        b_oi,
        b_funding,
        b_mark,
        b_spot,
        b_liq,
        b_ticks,
    ) = res

    m = b_ts_open.shape[0]
    bars: list[DollarBar] = []
    for idx in range(m):
        bars.append(
            DollarBar(
                instrument_id=instrument_id,
                horizon_label=horizon_label,
                ts_open_ns=int(b_ts_open[idx]),
                ts_close_ns=int(b_ts_close[idx]),
                open=float(b_open[idx]),
                high=float(b_high[idx]),
                low=float(b_low[idx]),
                close=float(b_close[idx]),
                volume=float(b_vol[idx]),
                dollar_volume=float(b_dvol[idx]),
                buy_dollar_volume=float(b_buy_dvol[idx]),
                sell_dollar_volume=float(b_sell_dvol[idx]),
                open_interest=float(b_oi[idx]),
                funding_rate_8h=float(b_funding[idx]),
                mark_price=float(b_mark[idx]),
                spot_price=float(b_spot[idx]),
                liquidation_volume_usd=float(b_liq[idx]),
                tick_count=int(b_ticks[idx]),
            )
        )
    return bars


class DynamicDollarBarAggregator:
    """Streaming tick-by-tick aggregator emitting Dynamic Dollar Bars at target horizons."""

    def __init__(
        self,
        instrument_id: str,
        horizon_label: Literal["15m", "1h", "4h"] = "15m",
        target_dollar_volume: float = 25_000_000.0,
        adaptive_threshold: bool = True,
    ) -> None:
        self.instrument_id = instrument_id
        self.horizon_label: Literal["15m", "1h", "4h"] = horizon_label
        self.initial_threshold = float(target_dollar_volume)
        self.current_threshold = float(target_dollar_volume)
        self.adaptive_threshold = adaptive_threshold
        self._target_ns_per_bar = (86_400.0 * 1e9) / float(HORIZON_BARS_PER_DAY[horizon_label])

        self._in_bar: bool = False
        self._ts_open_ns: int = 0
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._volume: float = 0.0
        self._dollar_volume: float = 0.0
        self._buy_dollar_volume: float = 0.0
        self._sell_dollar_volume: float = 0.0
        self._liquidation_volume_usd: float = 0.0
        self._tick_count: int = 0

        self._latest_oi: float = 100_000.0
        self._latest_funding_8h: float = 0.0001
        self._latest_mark_price: float = 0.0
        self._latest_spot_price: float = 0.0

    def reset(self) -> None:
        """Reset current bar accumulation state."""
        self._in_bar = False
        self._volume = 0.0
        self._dollar_volume = 0.0
        self._buy_dollar_volume = 0.0
        self._sell_dollar_volume = 0.0
        self._liquidation_volume_usd = 0.0
        self._tick_count = 0
        self.current_threshold = self.initial_threshold

    def update_derivatives_context(
        self,
        funding_rate_8h: float | None = None,
        open_interest: float | None = None,
        mark_price: float | None = None,
        spot_price: float | None = None,
        liquidation_volume_usd: float = 0.0,
    ) -> None:
        """Update auxiliary derivatives state (funding, OI, mark/spot basis, liquidations)."""
        if funding_rate_8h is not None:
            self._latest_funding_8h = float(funding_rate_8h)
        if open_interest is not None:
            self._latest_oi = max(float(open_interest), 0.0)
        if mark_price is not None and mark_price > 0.0:
            self._latest_mark_price = float(mark_price)
        if spot_price is not None and spot_price > 0.0:
            self._latest_spot_price = float(spot_price)
        if liquidation_volume_usd > 0.0:
            self._liquidation_volume_usd += float(liquidation_volume_usd)

    def on_trade_tick(
        self,
        price: float,
        size: float,
        ts_ns: int,
        is_buyer_maker: bool = False,
    ) -> DollarBar | None:
        """Process a single trade tick and return a completed DollarBar if threshold is reached."""
        if price <= 0.0 or size < 0.0:
            raise ValueError(f"Invalid tick price={price}, size={size}")

        dvol = price * size
        if not self._in_bar:
            self._in_bar = True
            self._ts_open_ns = ts_ns
            self._open = price
            self._high = price
            self._low = price
            self._volume = 0.0
            self._dollar_volume = 0.0
            self._buy_dollar_volume = 0.0
            self._sell_dollar_volume = 0.0
            self._tick_count = 0

        if price > self._high:
            self._high = price
        if price < self._low:
            self._low = price
        self._close = price
        self._volume += size
        self._dollar_volume += dvol
        if not is_buyer_maker:
            self._buy_dollar_volume += dvol
        else:
            self._sell_dollar_volume += dvol
        self._tick_count += 1

        if self._dollar_volume < self.current_threshold:
            return None

        mark_p = self._latest_mark_price if self._latest_mark_price > 0.0 else price
        spot_p = self._latest_spot_price if self._latest_spot_price > 0.0 else price

        bar = DollarBar(
            instrument_id=self.instrument_id,
            horizon_label=self.horizon_label,
            ts_open_ns=self._ts_open_ns,
            ts_close_ns=ts_ns,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            dollar_volume=self._dollar_volume,
            buy_dollar_volume=self._buy_dollar_volume,
            sell_dollar_volume=self._sell_dollar_volume,
            open_interest=self._latest_oi,
            funding_rate_8h=self._latest_funding_8h,
            mark_price=mark_p,
            spot_price=spot_p,
            liquidation_volume_usd=self._liquidation_volume_usd,
            tick_count=self._tick_count,
        )

        if self.adaptive_threshold:
            elapsed_ns = float(max(ts_ns - self._ts_open_ns, 1_000_000_000))
            implied = self._dollar_volume * (self._target_ns_per_bar / elapsed_ns)
            clipped = min(max(implied, self.initial_threshold * 0.25), self.initial_threshold * 4.0)
            self.current_threshold = 0.90 * self.current_threshold + 0.10 * clipped

        self._in_bar = False
        self._liquidation_volume_usd = 0.0
        return bar


class DerivativesFeatureActor(_NautilusDataActor):  # type: ignore[misc]
    """Native NautilusTrader DataActor for Dynamic Dollar Bars and Positioning Regime Signals.

    Supports both live/backtest streaming tick ingestion and vectorized Polars batch ingestion.
    """

    def __init__(
        self,
        config: DerivativesActorConfig | None = None,
        snapshot_callbacks: Sequence[Callable[[PositioningSignalSnapshot], None]] | None = None,
    ) -> None:
        resolved_cfg = config or DerivativesActorConfig()
        if _HAS_NAUTILUS_ACTOR:
            try:
                nt_cfg = _NautilusDataActorConfig(actor_id=resolved_cfg.actor_id)
                super().__init__(nt_cfg)
            except Exception:
                pass
        else:
            super().__init__(resolved_cfg)

        self.actor_config: DerivativesActorConfig = resolved_cfg
        self._aggregator = DynamicDollarBarAggregator(
            instrument_id=resolved_cfg.instrument_id,
            horizon_label=resolved_cfg.horizon_label,
            target_dollar_volume=resolved_cfg.target_dollar_volume,
            adaptive_threshold=resolved_cfg.adaptive_threshold,
        )
        self._feature_engine = PositioningFeatureEngine(
            default_d=resolved_cfg.default_ffd_d,
            zscore_window=resolved_cfg.zscore_window,
            max_history=resolved_cfg.max_bar_history,
        )
        self._emitted_bars: list[DollarBar] = []
        self._emitted_snapshots: list[PositioningSignalSnapshot] = []
        self._callbacks: list[Callable[[PositioningSignalSnapshot], None]] = (
            list(snapshot_callbacks) if snapshot_callbacks is not None else []
        )

    @property
    def feature_engine(self) -> PositioningFeatureEngine:
        """Return the underlying PositioningFeatureEngine."""
        return self._feature_engine

    @property
    def emitted_bars(self) -> list[DollarBar]:
        """Return a copy of all emitted DollarBars."""
        return list(self._emitted_bars)

    @property
    def emitted_snapshots(self) -> list[PositioningSignalSnapshot]:
        """Return a copy of all emitted PositioningSignalSnapshots."""
        return list(self._emitted_snapshots)

    @property
    def latest_snapshot(self) -> PositioningSignalSnapshot | None:
        """Return the most recently emitted PositioningSignalSnapshot, or None."""
        return self._emitted_snapshots[-1] if self._emitted_snapshots else None

    def register_snapshot_callback(
        self, callback: Callable[[PositioningSignalSnapshot], None]
    ) -> None:
        """Register a downstream consumer callback invoked on each new PositioningSignalSnapshot."""
        self._callbacks.append(callback)

    def calibrate_ffd(
        self,
        training_bars: Sequence[DollarBar],
        split_id: str = "split_0",
        tau: float = 1e-5,
        max_adf_pvalue: float = 0.01,
    ) -> tuple[FFDWeightsConfig, FFDWeightsConfig]:
        """Calibrate and freeze per-split FFD weights for close and OI on training bars."""
        return self._feature_engine.calibrate_on_split(
            training_bars=training_bars,
            split_id=split_id,
            tau=tau,
            max_adf_pvalue=max_adf_pvalue,
        )

    def on_derivatives_state(
        self,
        funding_rate_8h: float | None = None,
        open_interest: float | None = None,
        mark_price: float | None = None,
        spot_price: float | None = None,
        liquidation_volume_usd: float = 0.0,
    ) -> None:
        """Update latest 8h funding rate, Open Interest, mark/spot prices, and liquidation volume."""
        self._aggregator.update_derivatives_context(
            funding_rate_8h=funding_rate_8h,
            open_interest=open_interest,
            mark_price=mark_price,
            spot_price=spot_price,
            liquidation_volume_usd=liquidation_volume_usd,
        )

    def on_dollar_bar(self, bar: DollarBar) -> PositioningSignalSnapshot:
        """Process a completed DollarBar, update causal FFD features, and emit snapshot."""
        self._emitted_bars.append(bar)
        if len(self._emitted_bars) > self.actor_config.max_bar_history:
            self._emitted_bars = self._emitted_bars[-self.actor_config.max_bar_history :]

        snapshot = self._feature_engine.on_dollar_bar(bar)
        self._emitted_snapshots.append(snapshot)
        if len(self._emitted_snapshots) > self.actor_config.max_bar_history:
            self._emitted_snapshots = self._emitted_snapshots[
                -self.actor_config.max_bar_history :
            ]

        for cb in self._callbacks:
            cb(snapshot)
        return snapshot

    def on_trade_tick(
        self,
        price: float,
        size: float,
        ts_ns: int,
        is_buyer_maker: bool = False,
    ) -> PositioningSignalSnapshot | None:
        """Ingest a live/backtest trade tick; emits PositioningSignalSnapshot when a bar closes."""
        bar = self._aggregator.on_trade_tick(
            price=price,
            size=size,
            ts_ns=ts_ns,
            is_buyer_maker=is_buyer_maker,
        )
        if bar is None:
            return None
        return self.on_dollar_bar(bar)

    def ingest_polars_ticks(self, ticks_df: pl.DataFrame) -> list[PositioningSignalSnapshot]:
        """Vectorized Polars + Numba tick ingestion returning all resulting snapshots."""
        bars = aggregate_dollar_bars_polars(
            ticks_df=ticks_df,
            instrument_id=self.actor_config.instrument_id,
            horizon_label=self.actor_config.horizon_label,
            target_dollar_volume=self.actor_config.target_dollar_volume,
            adaptive_threshold=self.actor_config.adaptive_threshold,
        )
        if not bars:
            return []

        for b in bars:
            self._emitted_bars.append(b)
        if len(self._emitted_bars) > self.actor_config.max_bar_history:
            self._emitted_bars = self._emitted_bars[-self.actor_config.max_bar_history :]

        snapshots = self._feature_engine.transform_batch(self._emitted_bars)
        new_snapshots = snapshots[-len(bars) :]
        for snap in new_snapshots:
            self._emitted_snapshots.append(snap)
            for cb in self._callbacks:
                cb(snap)
        return new_snapshots
