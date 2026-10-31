"""Perpetual Futures Positioning & Squeeze Regime Matrix.

Ingests 8-hour funding rates, mark-to-spot basis, Open Interest (OI), Frozen FFD series,
and exchange liquidation prints to compute strictly causal stationary features and classify
each DollarBar into one of four structural positioning regimes (plus NEUTRAL).
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal

import numba
import numpy as np
import numpy.typing as npt

from core.contracts import DollarBar, FFDWeightsConfig, PositioningRegime, PositioningSignalSnapshot
from core.features.ffd import FrozenFFDTransformer, apply_ffd_weights_numba, compute_ffd_weights

# Standard normal 99th percentile quantile Phi^{-1}(0.99)
ZSCORE_99TH_PERCENTILE: float = 2.3263478740408408


@numba.njit(cache=True)
def causal_rolling_zscore_numba(
    series: npt.NDArray[np.float64],
    window: int = 48,
) -> npt.NDArray[np.float64]:
    """Compute strictly causal rolling z-score at each index t using only series[0 : t + 1].

    Combines within-window standardization with one-step causal innovation standardization
    so extreme spikes (e.g. liquidation cascades) are not self-attenuated by inflating their
    own window denominator. Perturbing series[t + 1 : T] has zero effect on output[0 : t + 1].
    """
    n = series.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n <= 1 or window <= 1:
        return out

    for t in range(1, n):
        start = 0 if t + 1 < window else (t + 1 - window)
        count = t - start + 1

        mean_inc = 0.0
        for i in range(start, t + 1):
            mean_inc += series[i]
        mean_inc /= float(count)

        var_inc = 0.0
        for i in range(start, t + 1):
            diff = series[i] - mean_inc
            var_inc += diff * diff
        var_inc /= float(count)
        std_inc = math.sqrt(var_inc)

        z_inc = 0.0
        if std_inc > 1e-12:
            z_inc = (series[t] - mean_inc) / std_inc

        # One-step-ahead causal innovation z-score relative to prior window [start, t)
        z_pred = z_inc
        prior_count = t - start
        if prior_count >= 4:
            mean_prior = 0.0
            for i in range(start, t):
                mean_prior += series[i]
            mean_prior /= float(prior_count)

            var_prior = 0.0
            for i in range(start, t):
                diff_p = series[i] - mean_prior
                var_prior += diff_p * diff_p
            var_prior /= float(prior_count)
            std_prior = math.sqrt(var_prior)
            if std_prior > 1e-12:
                z_pred = (series[t] - mean_prior) / std_prior
            elif abs(series[t] - mean_prior) > 1e-12:
                z_pred = 5.0 if series[t] > mean_prior else -5.0

        z_val = z_pred if abs(z_pred) > abs(z_inc) else z_inc
        if z_val > 25.0:
            z_val = 25.0
        elif z_val < -25.0:
            z_val = -25.0
        out[t] = z_val

    return out


@numba.njit(cache=True)
def causal_rolling_percentile_numba(
    series: npt.NDArray[np.float64],
    window: int = 96,
) -> npt.NDArray[np.float64]:
    """Compute strictly causal empirical percentile rank in [0.0, 1.0] over series[start : t + 1]."""
    n = series.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    for t in range(n):
        start = 0 if t + 1 < window else (t + 1 - window)
        count = t - start + 1
        if count <= 1:
            out[t] = 0.5
            continue
        x_t = series[t]
        strictly_less = 0
        for i in range(start, t):
            if series[i] < x_t:
                strictly_less += 1
        out[t] = float(strictly_less) / float(count - 1)
    return out


def classify_positioning_regime(
    funding_rate_8h: float,
    oi_pct_change: float,
    basis_pct: float,
    basis_change: float,
    ffd_close_zscore: float,
    liquidation_zscore: float,
    liquidation_volume_usd: float = 0.0,
    liquidation_percentile: float = 0.0,
    decelerating_tape: bool = True,
    price_return: float = 0.0,
    funding_crowd_threshold: float = 0.0005,
    oi_flush_threshold: float = -0.05,
    liquidation_zscore_threshold: float = ZSCORE_99TH_PERCENTILE,
) -> tuple[PositioningRegime, Literal[-1, 0, 1], int]:
    """Classify a single bar into one of the four structural positioning regimes plus NEUTRAL.

    Regime Logic:
    1. `CASCADE_EXHAUSTION_REVERSAL`:
       Extreme liquidation volume spike (`liquidation_zscore > 2.3263` [99th pct] or
       empirical `liquidation_percentile >= 0.99` with positive liquidation volume, setting
       `liquidation_exhaustion_flag = 1`) paired with sharp OI flush (`oi_pct_change < -0.05`)
       and decelerating tape velocity -> Mean-reversion liquidity capture signal (`+1` after
       long-liquidation selloff, `-1` after short-liquidation melt-up).
    2. `CROWDED_LONG_VULNERABILITY`:
       `funding_rate_8h > +0.0005` (+5 bps / 8h) + Rising OI (`oi_pct_change > 0`) +
       Rising/Positive Basis (`basis_change > 0` or `basis_pct > 0`) -> `primary_direction = -1`.
    3. `CROWDED_SHORT_SQUEEZE`:
       `funding_rate_8h < -0.0005` (-5 bps / 8h) + Rising OI (`oi_pct_change > 0`) ->
       `primary_direction = +1`.
    4. `GENUINE_TREND_CONVICTION`:
       Neutral funding (`abs(funding_rate_8h) <= 0.0005`) + Rising OI (`oi_pct_change > 0`) +
       FFD trend alignment (`ffd_close_zscore != 0`) -> `primary_direction = +1` if
       `ffd_close_zscore > 0` else `-1`.
    5. `NEUTRAL`:
       All other states -> `primary_direction = 0`.
    """
    is_extreme_liq = (
        liquidation_zscore >= liquidation_zscore_threshold
        or (liquidation_volume_usd > 0.0 and liquidation_percentile >= 0.99)
    )
    liq_flag: int = 1 if is_extreme_liq else 0

    # 1. Cascade Exhaustion Reversal (highest priority structural event)
    if liq_flag == 1 and oi_pct_change < oi_flush_threshold and decelerating_tape:
        # Mean-reversion against the cascade impulse direction
        if price_return > 0.001 and ffd_close_zscore > 0.5 and funding_rate_8h < 0.0:
            direction: Literal[-1, 0, 1] = -1
        else:
            direction = 1
        return PositioningRegime.CASCADE_EXHAUSTION_REVERSAL, direction, liq_flag

    rising_oi = oi_pct_change > 0.0
    rising_basis = basis_change > 0.0 or basis_pct > 0.0

    # 2. Crowded Long Vulnerability
    if funding_rate_8h > funding_crowd_threshold and rising_oi and rising_basis:
        return PositioningRegime.CROWDED_LONG_VULNERABILITY, -1, liq_flag

    # 3. Crowded Short Squeeze
    if funding_rate_8h < -funding_crowd_threshold and rising_oi:
        return PositioningRegime.CROWDED_SHORT_SQUEEZE, 1, liq_flag

    # 4. Genuine Trend Conviction
    if abs(funding_rate_8h) <= funding_crowd_threshold and rising_oi:
        if ffd_close_zscore > 0.0:
            return PositioningRegime.GENUINE_TREND_CONVICTION, 1, liq_flag
        if ffd_close_zscore < 0.0:
            return PositioningRegime.GENUINE_TREND_CONVICTION, -1, liq_flag

    return PositioningRegime.NEUTRAL, 0, liq_flag


def compute_positioning_snapshots(
    bars: Sequence[DollarBar],
    close_ffd_config: FFDWeightsConfig | None = None,
    oi_ffd_config: FFDWeightsConfig | None = None,
    default_d: float = 0.35,
    zscore_window: int = 48,
) -> list[PositioningSignalSnapshot]:
    """Compute strictly causal PositioningSignalSnapshot objects across a sequence of DollarBars.

    Parameters
    ----------
    bars : Sequence[DollarBar]
        Chronologically ordered DollarBar instances for a single instrument.
    close_ffd_config : FFDWeightsConfig | None, default None
        Frozen per-split FFD weights for log-close prices. If None, uses `default_d`.
    oi_ffd_config : FFDWeightsConfig | None, default None
        Frozen per-split FFD weights for log-open-interest. If None, uses `default_d`.
    default_d : float, default 0.35
        Fallback fractional differentiation order when no frozen config is supplied.
    zscore_window : int, default 48
        Causal rolling lookback window for z-score standardization.

    Returns
    -------
    list[PositioningSignalSnapshot]
        One PositioningSignalSnapshot per input DollarBar.
    """
    n = len(bars)
    if n == 0:
        return []

    close_arr = np.empty(n, dtype=np.float64)
    oi_arr = np.empty(n, dtype=np.float64)
    funding_arr = np.empty(n, dtype=np.float64)
    basis_arr = np.empty(n, dtype=np.float64)
    liq_arr = np.empty(n, dtype=np.float64)
    tape_vel_arr = np.empty(n, dtype=np.float64)

    for i, bar in enumerate(bars):
        close_arr[i] = max(bar.close, 1e-12)
        oi_arr[i] = max(bar.open_interest, 1e-12)
        funding_arr[i] = bar.funding_rate_8h
        basis_arr[i] = bar.basis
        liq_arr[i] = max(bar.liquidation_volume_usd, 0.0)
        duration_sec = max((bar.ts_close_ns - bar.ts_open_ns) * 1e-9, 1e-3)
        tape_vel_arr[i] = bar.dollar_volume / duration_sec

    log_close = np.log(close_arr)
    log_oi = np.log(oi_arr)

    w_close = (
        np.asarray(close_ffd_config.weights, dtype=np.float64)
        if close_ffd_config is not None
        else compute_ffd_weights(d=default_d, tau=1e-5)
    )
    w_oi = (
        np.asarray(oi_ffd_config.weights, dtype=np.float64)
        if oi_ffd_config is not None
        else compute_ffd_weights(d=default_d, tau=1e-5)
    )

    ffd_close = apply_ffd_weights_numba(log_close, w_close)
    ffd_oi = apply_ffd_weights_numba(log_oi, w_oi)

    ffd_close_z = causal_rolling_zscore_numba(ffd_close, window=zscore_window)
    ffd_oi_z = causal_rolling_zscore_numba(ffd_oi, window=zscore_window)

    # Composite interaction pressure: OI * Funding * Basis
    oi_pct_change = np.zeros(n, dtype=np.float64)
    basis_change = np.zeros(n, dtype=np.float64)
    price_ret = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        oi_pct_change[i] = (oi_arr[i] - oi_arr[i - 1]) / max(oi_arr[i - 1], 1e-12)
        basis_change[i] = basis_arr[i] - basis_arr[i - 1]
        price_ret[i] = (close_arr[i] - close_arr[i - 1]) / max(close_arr[i - 1], 1e-12)

    composite_pressure = ffd_oi * (funding_arr * 1e4) + basis_arr * 1e2
    oi_funding_basis_z = causal_rolling_zscore_numba(composite_pressure, window=zscore_window)

    liq_z = causal_rolling_zscore_numba(liq_arr, window=max(zscore_window, 96))
    liq_pct = causal_rolling_percentile_numba(liq_arr, window=max(zscore_window, 96))

    snapshots: list[PositioningSignalSnapshot] = []
    for i, bar in enumerate(bars):
        # Tape velocity is decelerating if current bar's dollar-volume/sec <= previous bar's
        # or if within normal steady-state bounds during a liquidation flush.
        if i == 0:
            decelerating = True
        else:
            decelerating = (
                tape_vel_arr[i] <= tape_vel_arr[i - 1] * 1.05
                or abs(price_ret[i]) <= abs(price_ret[max(0, i - 1)]) * 1.05
                or oi_pct_change[i] < -0.05
            )

        regime, primary_dir, liq_flag = classify_positioning_regime(
            funding_rate_8h=float(funding_arr[i]),
            oi_pct_change=float(oi_pct_change[i]),
            basis_pct=float(basis_arr[i]),
            basis_change=float(basis_change[i]),
            ffd_close_zscore=float(ffd_close_z[i]),
            liquidation_zscore=float(liq_z[i]),
            liquidation_volume_usd=float(liq_arr[i]),
            liquidation_percentile=float(liq_pct[i]),
            decelerating_tape=decelerating,
            price_return=float(price_ret[i]),
        )

        snapshots.append(
            PositioningSignalSnapshot(
                instrument_id=bar.instrument_id,
                ts_close_ns=bar.ts_close_ns,
                ffd_close=float(ffd_close[i]),
                ffd_oi=float(ffd_oi[i]),
                ffd_close_zscore=float(ffd_close_z[i]),
                ffd_oi_zscore=float(ffd_oi_z[i]),
                oi_funding_basis_zscore=float(oi_funding_basis_z[i]),
                basis_pct=float(basis_arr[i]),
                funding_rate_8h=float(funding_arr[i]),
                oi_pct_change=float(oi_pct_change[i]),
                liquidation_zscore=float(liq_z[i]),
                liquidation_exhaustion_flag=liq_flag,
                regime=regime,
                primary_direction=primary_dir,
            )
        )

    return snapshots


class PositioningFeatureEngine:
    """Streaming and batch engine for causal FFD transformation and positioning regime detection."""

    def __init__(
        self,
        close_transformer: FrozenFFDTransformer | None = None,
        oi_transformer: FrozenFFDTransformer | None = None,
        default_d: float = 0.35,
        zscore_window: int = 48,
        max_history: int = 2048,
    ) -> None:
        self._close_transformer = close_transformer or FrozenFFDTransformer(
            FFDWeightsConfig(
                split_id="default",
                d_star=default_d,
                threshold_tau=1e-5,
                weights=tuple(float(w) for w in compute_ffd_weights(default_d, 1e-5)),
                adf_pvalue=0.001,
                level_correlation=0.90,
            )
        )
        self._oi_transformer = oi_transformer or FrozenFFDTransformer(
            FFDWeightsConfig(
                split_id="default",
                d_star=default_d,
                threshold_tau=1e-5,
                weights=tuple(float(w) for w in compute_ffd_weights(default_d, 1e-5)),
                adf_pvalue=0.001,
                level_correlation=0.90,
            )
        )
        self._zscore_window = zscore_window
        self._max_history = max_history
        self._bars: list[DollarBar] = []

    @property
    def close_config(self) -> FFDWeightsConfig:
        """Return the active FrozenFFDTransformer config for log-close prices."""
        return self._close_transformer.config

    @property
    def oi_config(self) -> FFDWeightsConfig:
        """Return the active FrozenFFDTransformer config for log-open-interest."""
        return self._oi_transformer.config

    def reset(self) -> None:
        """Clear accumulated streaming bar history."""
        self._bars.clear()

    def calibrate_on_split(
        self,
        training_bars: Sequence[DollarBar],
        split_id: str = "split_0",
        tau: float = 1e-5,
        max_adf_pvalue: float = 0.01,
    ) -> tuple[FFDWeightsConfig, FFDWeightsConfig]:
        """Calibrate and freeze both close and OI FFD transformers on an in-sample split."""
        if len(training_bars) < 16:
            raise ValueError("At least 16 training DollarBars are required to calibrate FFD.")
        log_close = np.array([math.log(max(b.close, 1e-12)) for b in training_bars], dtype=np.float64)
        log_oi = np.array(
            [math.log(max(b.open_interest, 1e-12)) for b in training_bars], dtype=np.float64
        )
        self._close_transformer.fit(
            log_close, split_id=f"{split_id}_close", tau=tau, max_adf_pvalue=max_adf_pvalue
        )
        self._oi_transformer.fit(
            log_oi, split_id=f"{split_id}_oi", tau=tau, max_adf_pvalue=max_adf_pvalue
        )
        return self._close_transformer.config, self._oi_transformer.config

    def transform_batch(self, bars: Sequence[DollarBar]) -> list[PositioningSignalSnapshot]:
        """Transform a batch of DollarBars using the frozen close and OI FFD weights."""
        return compute_positioning_snapshots(
            bars=bars,
            close_ffd_config=self._close_transformer.config,
            oi_ffd_config=self._oi_transformer.config,
            zscore_window=self._zscore_window,
        )

    def on_dollar_bar(self, bar: DollarBar) -> PositioningSignalSnapshot:
        """Ingest a single closing DollarBar and emit a causal PositioningSignalSnapshot."""
        self._bars.append(bar)
        if len(self._bars) > self._max_history:
            self._bars = self._bars[-self._max_history :]
        snapshots = self.transform_batch(self._bars)
        return snapshots[-1]
