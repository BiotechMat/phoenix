"""Fee-Adjusted Triple-Barrier Labelling for medium-frequency Dynamic Dollar Bars.

Implements causal dollar-bar volatility estimation, hard minimum profit hurdles
(>= 80 bps and >= 6x round-trip fee + slippage), and exact deduction of round-trip
execution fees and accrued 8-hour perpetual funding costs before assigning
meta-labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from core.contracts import DollarBar, FeeSchedule, TripleBarrierConfig, TripleBarrierLabel

NANOSECONDS_PER_8H: float = 8.0 * 3600.0 * 1_000_000_000.0

HORIZON_TO_8H_FRACTION: dict[str, float] = {
    "15m": 0.25 / 8.0,
    "1h": 1.0 / 8.0,
    "4h": 4.0 / 8.0,
}


def _bar_parkinson_sigma(bar: DollarBar, min_sigma: float = 1e-4) -> float:
    """Compute single-bar Parkinson high-low range volatility estimator."""
    if bar.high <= bar.low or bar.low <= 0.0:
        return min_sigma
    log_hl = math.log(bar.high / bar.low)
    sigma_p = math.sqrt((log_hl * log_hl) / (4.0 * math.log(2.0)))
    return max(sigma_p, min_sigma)


def compute_causal_bar_volatility(
    bars: Sequence[DollarBar],
    window: int = 20,
    min_sigma: float = 1e-4,
) -> npt.NDArray[np.float64]:
    """Compute strictly causal rolling realized volatility sigma_bar for each DollarBar.

    For bar index t:
    - Uses only bars[0..t] (strictly causal, zero look-ahead).
    - Blends close-to-close log return standard deviation over `window` trailing bars
      with the Parkinson intra-bar range estimator for initial warmup bars (t < 2).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_sigma <= 0.0:
        raise ValueError(f"min_sigma must be > 0.0, got {min_sigma}")

    n_bars = len(bars)
    sigmas = np.empty(n_bars, dtype=np.float64)
    if n_bars == 0:
        return sigmas

    log_closes = np.array([math.log(b.close) for b in bars], dtype=np.float64)
    log_returns = np.zeros(n_bars, dtype=np.float64)
    if n_bars > 1:
        log_returns[1:] = np.diff(log_closes)

    for t in range(n_bars):
        parkinson_t = _bar_parkinson_sigma(bars[t], min_sigma=min_sigma)
        if t == 0:
            sigmas[t] = parkinson_t
        elif t == 1:
            ret_abs = abs(float(log_returns[1]))
            sigmas[t] = max(0.5 * (ret_abs + parkinson_t), min_sigma)
        else:
            start_idx = max(1, t - window + 1)
            window_rets = log_returns[start_idx : t + 1]
            std_ret = float(np.std(window_rets, ddof=1))
            # Combine realized close-to-close vol with a 20% Parkinson floor to avoid
            # zero-volatility degeneracy on flat close sequences with intra-bar range.
            sigmas[t] = max(std_ret, 0.2 * parkinson_t, min_sigma)

    return sigmas


def compute_effective_barriers(
    sigma_bar: float,
    config: TripleBarrierConfig,
) -> tuple[float, float]:
    """Compute fee-adjusted (upper_barrier, lower_barrier) in decimal fraction.

    Enforces the governing minimum profit hurdle:
        Barrier_min = max(k_pt * sigma_bar, 6 * (Fee_roundtrip + Slippage_est)) >= 80 bps
    """
    if sigma_bar < 0.0:
        raise ValueError(f"sigma_bar must be non-negative, got {sigma_bar}")

    upper_barrier = config.effective_profit_barrier_decimal(sigma_bar)
    fee_plus_slip_decimal = config.fee_schedule.roundtrip_maker_plus_slippage_bps * 1e-4
    lower_floor = max(2.0 * fee_plus_slip_decimal, 20.0 * 1e-4)
    lower_barrier = max(config.sl_sigma_mult * sigma_bar, lower_floor)
    return float(upper_barrier), float(lower_barrier)


def _compute_bar_8h_fraction(prev_bar: DollarBar, curr_bar: DollarBar) -> float:
    """Compute fraction of an 8-hour funding interval elapsed across `curr_bar`."""
    delta_ns = curr_bar.ts_close_ns - prev_bar.ts_close_ns
    if delta_ns <= 0:
        delta_ns = curr_bar.ts_close_ns - curr_bar.ts_open_ns

    # Use exact nanosecond delta when realistic timestamps (>= 1 second) are present.
    if delta_ns >= 1_000_000_000:
        return float(delta_ns) / NANOSECONDS_PER_8H

    return HORIZON_TO_8H_FRACTION.get(curr_bar.horizon_label, 0.25 / 8.0)


def label_single_bar_triple_barrier(
    bars: Sequence[DollarBar],
    bar_index: int,
    primary_direction: Literal[-1, 0, 1],
    sigma_bar: float,
    config: TripleBarrierConfig | None = None,
) -> TripleBarrierLabel:
    """Assign a net-of-fee and net-of-funding `TripleBarrierLabel` to `bars[bar_index]`.

    Deductions:
    - Exact round-trip fee + slippage (`config.fee_schedule.roundtrip_maker_plus_slippage_bps * 1e-4`).
    - Accrued 8-hour funding cost over each held bar (`direction * funding_rate_8h * dt_8h`).

    Meta-label assignment:
    - `meta_label = 1` if and only if `primary_direction != 0` and the trade either hits
      the upper profit barrier with positive net-of-fee/funding return clearing the
      fee hurdle or achieves a net-of-fee/funding return >= `upper_barrier` before
      hitting the stop-loss or vertical barrier.
    """
    cfg = config or TripleBarrierConfig()
    n_bars = len(bars)
    if not (0 <= bar_index < n_bars):
        raise IndexError(f"bar_index {bar_index} out of bounds for sequence of length {n_bars}")
    if primary_direction not in (-1, 0, 1):
        raise ValueError(f"primary_direction must be in {{-1, 0, 1}}, got {primary_direction}")

    upper_barrier, lower_barrier = compute_effective_barriers(sigma_bar, cfg)

    # Neutral primary signal or final bar in dataset -> no trade can be entered/held
    if primary_direction == 0 or bar_index == n_bars - 1:
        return TripleBarrierLabel(
            bar_index=bar_index,
            primary_direction=primary_direction,
            gross_return=0.0,
            net_return=0.0,
            fee_cost=0.0,
            funding_cost=0.0,
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            hit_bar_offset=0,
            barrier_touched="NONE",
            meta_label=0,
        )

    entry_bar = bars[bar_index]
    entry_price = entry_bar.close
    direction_f = float(primary_direction)
    fee_cost = cfg.fee_schedule.roundtrip_maker_plus_slippage_bps * 1e-4

    max_end_index = min(bar_index + cfg.vertical_barrier_bars, n_bars - 1)
    accrued_funding_cost = 0.0

    hit_offset = max_end_index - bar_index
    barrier_touched: Literal["PROFIT", "STOP", "VERTICAL", "NONE"] = "VERTICAL"
    realized_gross_return = 0.0

    for j in range(bar_index + 1, max_end_index + 1):
        curr_bar = bars[j]
        prev_bar = bars[j - 1]
        dt_8h = _compute_bar_8h_fraction(prev_bar, curr_bar)
        # Positive funding_rate_8h means longs (+1) pay shorts (-1)
        accrued_funding_cost += direction_f * curr_bar.funding_rate_8h * dt_8h

        if primary_direction == 1:
            fav_excursion = (curr_bar.high - entry_price) / entry_price
            adv_excursion = (curr_bar.low - entry_price) / entry_price
            close_return = (curr_bar.close - entry_price) / entry_price
        else:
            fav_excursion = (entry_price - curr_bar.low) / entry_price
            adv_excursion = (entry_price - curr_bar.high) / entry_price
            close_return = (entry_price - curr_bar.close) / entry_price

        hit_profit = fav_excursion >= upper_barrier
        hit_stop = adv_excursion <= -lower_barrier

        if hit_profit and hit_stop:
            # Disambiguate same-bar double touch using close return sign, defaulting
            # conservatively to STOP when close does not confirm profit barrier.
            if close_return >= upper_barrier:
                hit_stop = False
            else:
                hit_profit = False

        if hit_stop:
            hit_offset = j - bar_index
            barrier_touched = "STOP"
            realized_gross_return = min(close_return, -lower_barrier)
            break

        if hit_profit:
            hit_offset = j - bar_index
            barrier_touched = "PROFIT"
            realized_gross_return = max(close_return, upper_barrier)
            break

        if j == max_end_index:
            hit_offset = j - bar_index
            barrier_touched = "VERTICAL"
            realized_gross_return = close_return

    net_return = realized_gross_return - fee_cost - accrued_funding_cost
    min_net_hurdle = upper_barrier - fee_cost

    # A trade receives meta_label = 1 iff it cleared the profit hurdle net of fees & funding
    # before hitting STOP or decaying below hurdle at VERTICAL expiry.
    if (
        barrier_touched == "PROFIT"
        and net_return > 0.0
        and net_return >= min_net_hurdle - max(accrued_funding_cost, 0.0) - 1e-12
    ) or (barrier_touched == "VERTICAL" and net_return >= upper_barrier):
        meta_label: Literal[0, 1] = 1
    else:
        meta_label = 0

    return TripleBarrierLabel(
        bar_index=bar_index,
        primary_direction=primary_direction,
        gross_return=float(realized_gross_return),
        net_return=float(net_return),
        fee_cost=float(fee_cost),
        funding_cost=float(accrued_funding_cost),
        upper_barrier=float(upper_barrier),
        lower_barrier=float(lower_barrier),
        hit_bar_offset=int(hit_offset),
        barrier_touched=barrier_touched,
        meta_label=meta_label,
    )


def label_triple_barrier_series(
    bars: Sequence[DollarBar],
    primary_directions: Sequence[int] | npt.NDArray[np.int_],
    config: TripleBarrierConfig | None = None,
    sigma_bars: Sequence[float] | npt.NDArray[np.float64] | None = None,
    vol_window: int = 20,
) -> list[TripleBarrierLabel]:
    """Compute fee-adjusted `TripleBarrierLabel` outcomes across a sequence of `DollarBar`s."""
    n_bars = len(bars)
    if len(primary_directions) != n_bars:
        raise ValueError(
            f"Length mismatch: len(bars)={n_bars} != len(primary_directions)={len(primary_directions)}"
        )
    if n_bars == 0:
        return []

    cfg = config or TripleBarrierConfig()
    if sigma_bars is None:
        computed_sigmas = compute_causal_bar_volatility(bars, window=vol_window)
    else:
        if len(sigma_bars) != n_bars:
            raise ValueError(
                f"Length mismatch: len(bars)={n_bars} != len(sigma_bars)={len(sigma_bars)}"
            )
        computed_sigmas = np.asarray(sigma_bars, dtype=np.float64)

    labels: list[TripleBarrierLabel] = []
    for i in range(n_bars):
        raw_dir = int(primary_directions[i])
        if raw_dir not in (-1, 0, 1):
            raise ValueError(f"primary_directions[{i}] must be in {{-1, 0, 1}}, got {raw_dir}")
        typed_dir: Literal[-1, 0, 1] = -1 if raw_dir == -1 else (1 if raw_dir == 1 else 0)
        lbl = label_single_bar_triple_barrier(
            bars=bars,
            bar_index=i,
            primary_direction=typed_dir,
            sigma_bar=float(computed_sigmas[i]),
            config=cfg,
        )
        labels.append(lbl)

    return labels


class FeeAdjustedTripleBarrierLabeler:
    """Stateful helper for fee-adjusted Triple-Barrier labelling over Dynamic Dollar Bars."""

    def __init__(
        self,
        config: TripleBarrierConfig | None = None,
        vol_window: int = 20,
        min_sigma: float = 1e-4,
    ) -> None:
        self._config = config or TripleBarrierConfig()
        self._vol_window = vol_window
        self._min_sigma = min_sigma

    @property
    def config(self) -> TripleBarrierConfig:
        """Active `TripleBarrierConfig`."""
        return self._config

    @property
    def fee_schedule(self) -> FeeSchedule:
        """Active `FeeSchedule` from the barrier configuration."""
        return self._config.fee_schedule

    @property
    def vol_window(self) -> int:
        """Causal volatility lookback window in bars."""
        return self._vol_window

    def compute_volatility(self, bars: Sequence[DollarBar]) -> npt.NDArray[np.float64]:
        """Compute strictly causal realized bar volatility array."""
        return compute_causal_bar_volatility(
            bars=bars,
            window=self._vol_window,
            min_sigma=self._min_sigma,
        )

    def compute_barriers(self, sigma_bar: float) -> tuple[float, float]:
        """Return `(upper_barrier, lower_barrier)` in decimal fraction for `sigma_bar`."""
        return compute_effective_barriers(sigma_bar=sigma_bar, config=self._config)

    def label_bar(
        self,
        bars: Sequence[DollarBar],
        bar_index: int,
        primary_direction: Literal[-1, 0, 1],
        sigma_bar: float | None = None,
    ) -> TripleBarrierLabel:
        """Label a single bar index with automatic causal volatility fallback."""
        if sigma_bar is None:
            causal_prefix = bars[: bar_index + 1]
            sigmas = self.compute_volatility(causal_prefix)
            eff_sigma = float(sigmas[-1])
        else:
            eff_sigma = float(sigma_bar)

        return label_single_bar_triple_barrier(
            bars=bars,
            bar_index=bar_index,
            primary_direction=primary_direction,
            sigma_bar=eff_sigma,
            config=self._config,
        )

    def label_bars(
        self,
        bars: Sequence[DollarBar],
        primary_directions: Sequence[int] | npt.NDArray[np.int_],
        sigma_bars: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> list[TripleBarrierLabel]:
        """Label a full series of `DollarBar`s."""
        return label_triple_barrier_series(
            bars=bars,
            primary_directions=primary_directions,
            config=self._config,
            sigma_bars=sigma_bars,
            vol_window=self._vol_window,
        )


__all__ = [
    "NANOSECONDS_PER_8H",
    "FeeAdjustedTripleBarrierLabeler",
    "compute_causal_bar_volatility",
    "compute_effective_barriers",
    "label_single_bar_triple_barrier",
    "label_triple_barrier_series",
]
