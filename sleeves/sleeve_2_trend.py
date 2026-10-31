"""Sleeve 2: time-series trend on 1h bars (REINFORCEMENT_PLAN task P2.2). Default off.

Per asset and lookback ``L`` in {24h, 168h, 720h}:

    s_L = tanh( log(P_t / P_{t-L}) / (sigma_1h(L) * sqrt(L)) )

where ``sigma_1h(L)`` is the standard deviation of hourly log returns over the lookback. The forecast is
the equal-weighted mean of ``s_L`` (bounded in [-1, 1]). Position sizing is inverse-volatility on a
25-day EWMA of hourly returns, targeting ``target_vol_annual`` per asset, divided by the number of
assets and multiplied by the diversification adjustment ``1 / sqrt(1 + (N - 1) * rho_bar)`` with
``rho_bar`` the average pairwise correlation over the last 30 days (floored at 0). Weights are capped
per asset, and a no-trade buffer suppresses rebalances smaller than ``rebalance_buffer``.

Everything is causal: the weight for hour ``t`` uses closes up to and including ``t`` and is held over
``t -> t+1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import warnings
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from sleeves.sleeve_3_funding_carry import ALO_ROUTING_TAG, SleeveSignalOutput

HOURS_PER_YEAR: Final[float] = 8760.0
TREND_SLEEVE_ID: Final[str] = "trend"


class TrendSleeveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False, description="Ships off; enable only after CPCV and shadow mode")
    shadow: bool = Field(default=True, description="When enabled, trade in the shadow book first (P5.1); False trades live")
    lookbacks_hours: tuple[int, ...] = (24, 168, 720)
    vol_ewma_span_hours: int = Field(default=25 * 24, ge=2)
    target_vol_annual: float = Field(default=0.20, gt=0.0)
    max_weight_per_asset: float = Field(default=0.5, gt=0.0, le=2.5)
    corr_window_hours: int = Field(default=720, ge=24)
    rebalance_buffer: float = Field(default=0.05, ge=0.0, description="Minimum |change| in weight that triggers a trade")

    @property
    def min_history_hours(self) -> int:
        return max(max(self.lookbacks_hours), self.corr_window_hours) + 1


class TrendSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: str
    forecast: float = Field(ge=-1.0, le=1.0)
    forecast_by_lookback: dict[int, float]
    vol_annual: float = Field(ge=0.0)
    corr_adjustment: float = Field(gt=0.0, le=1.0)
    target_weight: float

    def to_sleeve_signal_output(self) -> SleeveSignalOutput:
        direction = 1 if self.target_weight > 0.0 else (-1 if self.target_weight < 0.0 else 0)
        return SleeveSignalOutput(
            sleeve_id=TREND_SLEEVE_ID,
            target_asset=self.asset,
            direction=direction,
            # A confidence proxy only; trend sizing comes from target_weight, not from a Kelly fraction.
            meta_probability=float(min(1.0, 0.5 + 0.5 * abs(self.forecast))),
            raw_expected_return=0.0,
            recommended_leverage=abs(self.target_weight),
            routing_tag=ALO_ROUTING_TAG,
            forensic_metadata={
                "forecast": self.forecast,
                "forecast_by_lookback": self.forecast_by_lookback,
                "vol_annual": self.vol_annual,
                "corr_adjustment": self.corr_adjustment,
                "target_weight": self.target_weight,
            },
        )


# ------------------------------------------------------------------------------------------------
# Vectorised building blocks
# ------------------------------------------------------------------------------------------------


def _rolling_sum(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Sum over the trailing ``window`` values ending at each index (NaN before a full window)."""
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    if window <= 0 or x.shape[0] < window:
        return out
    c = np.concatenate(([0.0], np.cumsum(x)))
    out[window - 1 :] = c[window:] - c[:-window]
    return out


def lookback_forecasts(log_closes: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """``tanh(log return over L / (std of hourly log returns over L * sqrt(L)))`` at every index."""
    lc = np.asarray(log_closes, dtype=np.float64)
    n = lc.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= lookback:
        return out
    r1 = np.concatenate(([np.nan], np.diff(lc)))
    r1z = np.where(np.isfinite(r1), r1, 0.0)
    s1 = _rolling_sum(r1z, lookback)
    s2 = _rolling_sum(r1z * r1z, lookback)
    var = (s2 - s1 * s1 / lookback) / max(lookback - 1, 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    ret_l = np.full(n, np.nan, dtype=np.float64)
    ret_l[lookback:] = lc[lookback:] - lc[:-lookback]
    denom = sd * math.sqrt(lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 1e-12, ret_l / denom, 0.0)
    out[lookback:] = np.tanh(z[lookback:])
    return out


def ewma_vol_annual(log_closes: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    """Annualised EWMA volatility of hourly log returns (causal)."""
    lc = np.asarray(log_closes, dtype=np.float64)
    n = lc.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    alpha = 2.0 / (span + 1.0)
    r = np.diff(lc)
    var = float(r[0] * r[0])
    out[1] = math.sqrt(var * HOURS_PER_YEAR)
    for t in range(1, r.shape[0]):
        var = (1.0 - alpha) * var + alpha * float(r[t] * r[t])
        out[t + 1] = math.sqrt(var * HOURS_PER_YEAR)
    return out


def rolling_average_pairwise_correlation(log_closes: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Average off-diagonal correlation of hourly log returns over the trailing window, per index.

    ``log_closes`` has shape ``(T, N)``.
    """
    lc = np.asarray(log_closes, dtype=np.float64)
    t_len, n_assets = lc.shape
    out = np.full(t_len, np.nan, dtype=np.float64)
    if n_assets < 2 or t_len <= window:
        return out
    r = np.vstack([np.full((1, n_assets), 0.0), np.diff(lc, axis=0)])
    sums = [_rolling_sum(r[:, i], window) for i in range(n_assets)]
    sq = [_rolling_sum(r[:, i] * r[:, i], window) for i in range(n_assets)]
    corrs: list[NDArray[np.float64]] = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            sxy = _rolling_sum(r[:, i] * r[:, j], window)
            cov = sxy - sums[i] * sums[j] / window
            vi = sq[i] - sums[i] ** 2 / window
            vj = sq[j] - sums[j] ** 2 / window
            with np.errstate(divide="ignore", invalid="ignore"):
                corrs.append(np.where((vi > 1e-18) & (vj > 1e-18), cov / np.sqrt(vi * vj), np.nan))
    stacked = np.vstack(corrs)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns before a full window
        out = np.nanmean(stacked, axis=0)
    return out


class TrendBacktest(BaseModel):
    """Aligned hourly series for a multi-asset trend backtest."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    assets: tuple[str, ...]
    ts_ms: NDArray[np.int64]
    weights: NDArray[np.float64] = Field(description="(T, N) weight held over t -> t+1")
    aims: NDArray[np.float64] = Field(description="(T, N) capped target weight before the rebalance policy")
    forecasts: NDArray[np.float64]
    simple_returns_next: NDArray[np.float64] = Field(description="(T, N) close[t+1] / close[t] - 1; NaN at the end")


def backtest_trend(
    ts_ms: Sequence[int] | NDArray[np.int64],
    closes: Mapping[str, Sequence[float] | NDArray[np.float64]],
    config: TrendSleeveConfig | None = None,
) -> TrendBacktest:
    """Causal weights for every hour from aligned closes (``closes[asset]`` all the same length)."""
    cfg = config or TrendSleeveConfig()
    assets = tuple(closes.keys())
    lc = np.column_stack([np.log(np.asarray(closes[a], dtype=np.float64)) for a in assets])
    t_len, n_assets = lc.shape

    forecasts = np.zeros((t_len, n_assets), dtype=np.float64)
    vols = np.zeros((t_len, n_assets), dtype=np.float64)
    for j in range(n_assets):
        per_lb = np.vstack([lookback_forecasts(lc[:, j], lb) for lb in cfg.lookbacks_hours])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns before the longest lookback
            f = np.nanmean(per_lb, axis=0)
        f = np.where(np.all(np.isfinite(per_lb), axis=0), f, np.nan)
        forecasts[:, j] = f
        vols[:, j] = ewma_vol_annual(lc[:, j], cfg.vol_ewma_span_hours)

    rho = rolling_average_pairwise_correlation(lc, cfg.corr_window_hours)
    rho_f = np.where(np.isfinite(rho), np.maximum(rho, 0.0), np.nan)
    corr_adj = 1.0 / np.sqrt(1.0 + (n_assets - 1) * rho_f)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = forecasts * (cfg.target_vol_annual / vols) / n_assets * corr_adj[:, None]
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw[: cfg.min_history_hours] = 0.0
    capped = np.clip(raw, -cfg.max_weight_per_asset, cfg.max_weight_per_asset)

    weights = np.zeros_like(capped)
    current = np.zeros(n_assets, dtype=np.float64)
    for t in range(t_len):
        move = np.abs(capped[t] - current) > cfg.rebalance_buffer
        flat_target = capped[t] == 0.0
        current = np.where(move | flat_target, capped[t], current)
        weights[t] = current

    closes_arr = np.exp(lc)
    nxt = np.full_like(closes_arr, np.nan)
    nxt[:-1] = closes_arr[1:] / closes_arr[:-1] - 1.0
    return TrendBacktest(
        assets=assets,
        ts_ms=np.asarray(ts_ms, dtype=np.int64),
        weights=weights,
        aims=capped,
        forecasts=np.where(np.isfinite(forecasts), forecasts, 0.0),
        simple_returns_next=nxt,
    )


# ------------------------------------------------------------------------------------------------
# Live evaluator
# ------------------------------------------------------------------------------------------------


class TrendSleeve:
    """Evaluates current trend targets from aligned hourly closes; the same maths as the backtest."""

    def __init__(self, config: TrendSleeveConfig | None = None) -> None:
        self.config: TrendSleeveConfig = config or TrendSleeveConfig()
        self._last_weights: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def evaluate(self, hourly_closes: Mapping[str, Sequence[float]]) -> dict[str, TrendSignal]:
        """Signals for every asset; empty when any asset lacks ``min_history_hours`` closes."""
        cfg = self.config
        lengths = {a: len(v) for a, v in hourly_closes.items()}
        if not lengths or min(lengths.values()) < cfg.min_history_hours:
            return {}
        n = min(lengths.values())
        aligned = {a: list(v)[-n:] for a, v in hourly_closes.items()}
        bt = backtest_trend(np.arange(n, dtype=np.int64), aligned, cfg)
        lc = np.column_stack([np.log(np.asarray(aligned[a], dtype=np.float64)) for a in bt.assets])
        rho = rolling_average_pairwise_correlation(lc, cfg.corr_window_hours)[-1]
        corr_adj = float(1.0 / math.sqrt(1.0 + (len(bt.assets) - 1) * max(0.0, float(rho) if math.isfinite(float(rho)) else 0.0)))
        out: dict[str, TrendSignal] = {}
        for j, asset in enumerate(bt.assets):
            per_lb = {lb: float(lookback_forecasts(lc[:, j], lb)[-1]) for lb in cfg.lookbacks_hours}
            vol = float(ewma_vol_annual(lc[:, j], cfg.vol_ewma_span_hours)[-1])
            out[asset] = TrendSignal(
                asset=asset,
                forecast=float(np.clip(bt.forecasts[-1, j], -1.0, 1.0)),
                forecast_by_lookback=per_lb,
                vol_annual=vol if math.isfinite(vol) else 0.0,
                corr_adjustment=min(1.0, max(corr_adj, 1e-6)),
                target_weight=float(bt.weights[-1, j]),
            )
        self._last_weights = {a: s.target_weight for a, s in out.items()}
        return out


__all__ = [
    "TREND_SLEEVE_ID",
    "TrendBacktest",
    "TrendSignal",
    "TrendSleeve",
    "TrendSleeveConfig",
    "backtest_trend",
    "ewma_vol_annual",
    "lookback_forecasts",
    "rolling_average_pairwise_correlation",
]
