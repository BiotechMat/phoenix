"""Net-of-fee 45-split Combinatorial Purged Cross-Validation (CPCV) and Champion/Challenger Gatekeeper.

Enforces macOS 'spawn' multiprocessing and OpenMP/Polars thread caps (4 threads per worker,
max 2 parallel workers = 8 threads total on Apple M1 Max 8 P-cores) prior to numerical imports.
"""

from __future__ import annotations

import multiprocessing
import os

# Enforce macOS spawn isolation and 4-thread caps BEFORE importing numerical libraries.
multiprocessing.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Callable, Final, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from core.contracts import (
    CPCVValidationResult,
    ChampionChallengerState,
    ExecutionRegimeMode,
    FeeSchedule,
)

OMP_THREADS_PER_WORKER: Final[int] = 4
MAX_CPCV_WORKERS: Final[int] = 2
DEFAULT_N_GROUPS: Final[int] = 10
DEFAULT_K_TEST_GROUPS: Final[int] = 2
DEFAULT_EMBARGO_PCT: Final[float] = 0.01
PSR_PROMOTION_THRESHOLD: Final[float] = 0.95
DSR_PROMOTION_THRESHOLD: Final[float] = 0.95
PBO_MAX_THRESHOLD: Final[float] = 0.30
MIN_EXPECTED_NET_PNL_BPS: Final[float] = 15.0
LIVE_30D_PSR_FALLBACK_THRESHOLD: Final[float] = 0.80
MAX_CONSECUTIVE_FAILED_RETRAINS: Final[int] = 3


def _init_cpcv_worker(omp_threads: int = OMP_THREADS_PER_WORKER) -> None:
    """Initializer for spawned CPCV worker processes enforcing thread caps."""
    thread_str = str(omp_threads)
    os.environ["OMP_NUM_THREADS"] = thread_str
    os.environ["MKL_NUM_THREADS"] = thread_str
    os.environ["OPENBLAS_NUM_THREADS"] = thread_str
    os.environ["POLARS_MAX_THREADS"] = thread_str


@dataclass(frozen=True)
class CPCVSplit:
    """Single Combinatorial Purged Cross-Validation split specification."""

    split_id: int
    test_groups: tuple[int, ...]
    train_groups: tuple[int, ...]
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    purged_indices: NDArray[np.int64]
    embargoed_indices: NDArray[np.int64]
    group_test_slices: dict[int, NDArray[np.int64]]
    oos_path_map: dict[int, int]


@dataclass(frozen=True)
class NetReturnBreakdown:
    """Net-of-fee return series and transaction/funding cost attribution."""

    gross_returns: NDArray[np.float64]
    net_returns: NDArray[np.float64]
    fee_and_slippage_costs: NDArray[np.float64]
    funding_costs: NDArray[np.float64]
    turnover: NDArray[np.float64]
    expected_net_pnl_bps: float
    trade_count: int


def compute_net_of_fee_returns(
    gross_asset_returns: NDArray[np.float64],
    positions: NDArray[np.float64],
    funding_rates_8h: NDArray[np.float64] | float = 0.0,
    fee_schedule: FeeSchedule | None = None,
    bars_per_8h: float = 32.0,
    taker_fill_ratio: float = 0.0,
) -> NetReturnBreakdown:
    """Deduct exchange maker/taker fees, queue/spread slippage, and 8h funding rates.

    Parameters
    ----------
    gross_asset_returns:
        1D array of raw asset returns r_t per bar.
    positions:
        1D array of signed target positions pos_t in [-1, 1] (or {-1, 0, +1}) held over bar t.
    funding_rates_8h:
        8-hour perpetual funding rate as decimal fraction (e.g., 0.0001 = 1 bp per 8h).
        Positive funding means longs pay shorts.
    fee_schedule:
        Exchange fee schedule (defaults to standard USDT perpetual: 2.0 bps maker, 5.5 bps taker,
        1.0 bp slippage per leg).
    bars_per_8h:
        Number of bars per 8-hour funding interval (e.g., 32 for 15-minute bars).
    taker_fill_ratio:
        Fraction of volume executed as taker in [0.0, 1.0]. Defaults to 0.0 (Maker-First GTX).
    """
    r_asset = np.asarray(gross_asset_returns, dtype=np.float64).ravel()
    pos = np.asarray(positions, dtype=np.float64).ravel()
    if r_asset.shape[0] != pos.shape[0]:
        raise ValueError(
            f"Length mismatch between gross_asset_returns ({r_asset.shape[0]}) and positions ({pos.shape[0]})"
        )
    if not (0.0 <= taker_fill_ratio <= 1.0):
        raise ValueError(f"taker_fill_ratio must be in [0.0, 1.0], got {taker_fill_ratio}")
    if bars_per_8h <= 0.0:
        raise ValueError(f"bars_per_8h must be > 0.0, got {bars_per_8h}")

    schedule = fee_schedule or FeeSchedule.usdt_perp()
    n_bars = r_asset.shape[0]
    if n_bars == 0:
        empty = np.empty(0, dtype=np.float64)
        return NetReturnBreakdown(
            gross_returns=empty,
            net_returns=empty,
            fee_and_slippage_costs=empty,
            funding_costs=empty,
            turnover=empty,
            expected_net_pnl_bps=0.0,
            trade_count=0,
        )

    if isinstance(funding_rates_8h, (int, float)):
        funding_arr = np.full(n_bars, float(funding_rates_8h), dtype=np.float64)
    else:
        funding_arr = np.asarray(funding_rates_8h, dtype=np.float64).ravel()
        if funding_arr.shape[0] != n_bars:
            raise ValueError(
                f"Length mismatch between funding_rates_8h ({funding_arr.shape[0]}) and bars ({n_bars})"
            )

    prev_pos = np.concatenate([np.zeros(1, dtype=np.float64), pos[:-1]])
    turnover = np.abs(pos - prev_pos)

    effective_fee_rate = (
        (1.0 - taker_fill_ratio) * schedule.maker_rate
        + taker_fill_ratio * schedule.taker_rate
        + schedule.estimated_slippage_bps * 1e-4
    )
    fee_and_slippage_costs = turnover * effective_fee_rate
    funding_costs = pos * (funding_arr / bars_per_8h)

    gross_strat_returns = pos * r_asset
    net_returns = gross_strat_returns - fee_and_slippage_costs - funding_costs

    # Compute expected net PnL per discrete trade episode in basis points
    expected_net_pnl_bps, trade_count = _compute_expected_trade_net_pnl_bps(net_returns, pos)

    return NetReturnBreakdown(
        gross_returns=gross_strat_returns,
        net_returns=net_returns,
        fee_and_slippage_costs=fee_and_slippage_costs,
        funding_costs=funding_costs,
        turnover=turnover,
        expected_net_pnl_bps=expected_net_pnl_bps,
        trade_count=trade_count,
    )


def _compute_expected_trade_net_pnl_bps(
    net_returns: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
) -> tuple[float, int]:
    """Compute average net PnL per trade (in bps) across discrete active position episodes."""
    net = np.asarray(net_returns, dtype=np.float64).ravel()
    if net.size == 0:
        return 0.0, 0

    if positions is None:
        active_mask = np.abs(net) > 1e-12
        if not np.any(active_mask):
            return 0.0, 0
        return float(np.mean(net[active_mask]) * 1e4), int(np.sum(active_mask))

    pos = np.asarray(positions, dtype=np.float64).ravel()
    active_indices = np.flatnonzero(np.abs(pos) > 1e-9)
    if active_indices.size == 0:
        return 0.0, 0

    # Group contiguous active bars with identical sign into discrete trades
    trade_pnls: list[float] = []
    current_pnl = float(net[active_indices[0]])
    prev_idx = int(active_indices[0])
    prev_sign = float(np.sign(pos[prev_idx]))

    for idx_raw in active_indices[1:]:
        idx = int(idx_raw)
        sign_now = float(np.sign(pos[idx]))
        if idx == prev_idx + 1 and sign_now == prev_sign:
            current_pnl += float(net[idx])
        else:
            trade_pnls.append(current_pnl)
            current_pnl = float(net[idx])
        prev_idx = idx
        prev_sign = sign_now
    trade_pnls.append(current_pnl)

    mean_trade_bps = float(np.mean(trade_pnls) * 1e4)
    return mean_trade_bps, len(trade_pnls)


def _compute_sample_moments(returns: NDArray[np.float64]) -> tuple[float, float, float]:
    """Compute non-annualized sample Sharpe ratio, sample skewness (gamma_3), and Pearson kurtosis (gamma_4)."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return 0.0, 0.0, 3.0

    std = float(np.std(arr, ddof=1))
    if std <= 1e-15:
        return 0.0, 0.0, 3.0

    mean = float(np.mean(arr))
    sr = mean / std
    skewness = float(stats.skew(arr, bias=False)) if n >= 3 else 0.0
    kurtosis_pearson = float(stats.kurtosis(arr, fisher=False, bias=False)) if n >= 4 else 3.0
    if not math.isfinite(skewness):
        skewness = 0.0
    if not math.isfinite(kurtosis_pearson) or kurtosis_pearson <= 0.0:
        kurtosis_pearson = 3.0
    return sr, skewness, kurtosis_pearson


def compute_psr(
    returns: NDArray[np.float64] | Sequence[float],
    sr_benchmark: float = 0.0,
) -> float:
    """Compute Probabilistic Sharpe Ratio PSR(SR*) accounting for skewness and kurtosis.

    Formula (Bailey & Lopez de Prado, 2012):
        PSR(SR*) = Z( ((SR_hat - SR*) * sqrt(T - 1)) / sqrt(1 - gamma_3 * SR_hat + ((gamma_4 - 1) / 4) * SR_hat^2) )
    where gamma_3 is sample skewness and gamma_4 is sample Pearson (non-excess) kurtosis.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n_obs = arr.size
    if n_obs < 3:
        return 0.0

    sr_hat, gamma_3, gamma_4 = _compute_sample_moments(arr)
    if sr_hat == 0.0 and float( np.std(arr, ddof=1) ) <= 1e-15:
        return 0.0 if sr_benchmark >= 0.0 else 1.0

    variance_term = 1.0 - gamma_3 * sr_hat + ((gamma_4 - 1.0) / 4.0) * (sr_hat**2)
    denom = math.sqrt(max(variance_term, 1e-12))
    z_score = ((sr_hat - sr_benchmark) * math.sqrt(float(n_obs - 1))) / denom
    psr_val = float(stats.norm.cdf(z_score))
    return float(min(max(psr_val, 0.0), 1.0))


def compute_expected_max_sharpe(
    total_trials: int,
    n_obs: int,
    trial_sharpe_std: float | None = None,
) -> float:
    """Compute the expected maximum Sharpe ratio hurdle SR_0(S) across S independent trials.

    Uses the False Strategy Theorem (Bailey & Lopez de Prado, 2014):
        SR_0(S) = sigma_SR * ((1 - gamma_EM) * Z^-1(1 - 1/S) + gamma_EM * Z^-1(1 - 1/(S * e)))
    where sigma_SR = 1 / sqrt(T - 1) under the null hypothesis of per-observation returns.
    """
    if total_trials <= 1 or n_obs <= 1:
        return 0.0

    sigma_sr = (
        trial_sharpe_std
        if (trial_sharpe_std is not None and trial_sharpe_std > 0.0)
        else (1.0 / math.sqrt(float(n_obs - 1)))
    )
    euler_gamma = float(np.euler_gamma)
    s_float = float(total_trials)
    q1 = float(stats.norm.ppf(1.0 - 1.0 / s_float))
    q2 = float(stats.norm.ppf(1.0 - 1.0 / (s_float * math.e)))
    sr_0 = sigma_sr * ((1.0 - euler_gamma) * q1 + euler_gamma * q2)
    return float(max(sr_0, 0.0))


def compute_dsr(
    returns: NDArray[np.float64] | Sequence[float],
    total_trials: int = 45,
    trial_sharpes: NDArray[np.float64] | Sequence[float] | None = None,
) -> float:
    """Compute Deflated Sharpe Ratio (DSR) correcting for S strategy trials, skewness, and kurtosis."""
    if total_trials < 1:
        raise ValueError(f"total_trials must be >= 1, got {total_trials}")

    arr = np.asarray(returns, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n_obs = arr.size
    if n_obs < 3:
        return 0.0

    trial_std: float | None = None
    if trial_sharpes is not None:
        ts_arr = np.asarray(trial_sharpes, dtype=np.float64).ravel()
        ts_arr = ts_arr[np.isfinite(ts_arr)]
        if ts_arr.size >= 2:
            raw_std = float(np.std(ts_arr, ddof=1))
            null_std = 1.0 / math.sqrt(float(n_obs - 1))
            # Guard against annualized vs per-bar scale mismatch while preserving cross-trial dispersion
            trial_std = min(raw_std, 3.0 * null_std) if raw_std > 0.0 else null_std

    sr_0 = compute_expected_max_sharpe(
        total_trials=total_trials,
        n_obs=n_obs,
        trial_sharpe_std=trial_std,
    )
    return compute_psr(arr, sr_benchmark=sr_0)


def compute_pbo_logits(
    is_performance: NDArray[np.float64] | Sequence[float],
    oos_performance: NDArray[np.float64] | Sequence[float],
) -> NDArray[np.float64]:
    """Compute the 45 IS/OOS rank logits lambda_c = ln(omega_c / (1 - omega_c)).

    Supports both:
    - 2D matrices of shape (n_splits, n_strategies) where n_strategies >= 2:
      For each split c, finds the in-sample optimal strategy m*_c = argmax_m IS[c, m],
      computes its relative out-of-sample rank omega_c = rank(OOS[c, m*_c]) / (M + 1) in (0, 1),
      and returns lambda_c = ln(omega_c / (1 - omega_c)).
    - 1D vectors of shape (n_splits,) representing a single Challenger vs the zero/flat hurdle:
      Evaluates each split c against the flat/null alternative and cross-split distribution
      to produce omega_c in (0, 1) and lambda_c = ln(omega_c / (1 - omega_c)).
    """
    is_arr = np.asarray(is_performance, dtype=np.float64)
    oos_arr = np.asarray(oos_performance, dtype=np.float64)
    if is_arr.shape != oos_arr.shape:
        raise ValueError(
            f"is_performance shape {is_arr.shape} must match oos_performance shape {oos_arr.shape}"
        )

    if is_arr.ndim == 2 and is_arr.shape[1] >= 2:
        n_splits, n_strats = is_arr.shape
        logits = np.empty(n_splits, dtype=np.float64)
        for c in range(n_splits):
            best_is_idx = int(np.argmax(is_arr[c]))
            oos_ranks = stats.rankdata(oos_arr[c], method="average")
            omega_c = float(oos_ranks[best_is_idx]) / float(n_strats + 1)
            omega_c = min(max(omega_c, 1e-6), 1.0 - 1e-6)
            logits[c] = math.log(omega_c / (1.0 - omega_c))
        return logits

    is_1d = is_arr.ravel()
    oos_1d = oos_arr.ravel()
    n_splits = is_1d.size
    if n_splits == 0:
        return np.empty(0, dtype=np.float64)

    oos_scale = max(float(np.std(oos_1d, ddof=1)) if n_splits > 1 else 1.0, 1e-6)
    min_omega = 1.0 / float(n_splits + 1)
    max_omega = float(n_splits) / float(n_splits + 1)

    logits = np.empty(n_splits, dtype=np.float64)
    for c in range(n_splits):
        # If IS selected an active edge, evaluate whether OOS preserved positive performance
        signed_oos = oos_1d[c] if is_1d[c] >= 0.0 else -oos_1d[c]
        omega_c = 0.5 + 0.45 * math.tanh(float(signed_oos) / oos_scale)
        omega_c = min(max(omega_c, min_omega), max_omega)
        logits[c] = math.log(omega_c / (1.0 - omega_c))
    return logits


def compute_pbo(
    is_performance: NDArray[np.float64] | Sequence[float],
    oos_performance: NDArray[np.float64] | Sequence[float],
) -> float:
    """Compute Probability of Backtest Overfitting (PBO) across the 45 IS/OOS rank logits.

    PBO = (1 / C) * sum_{c=1}^C I(lambda_c <= 0), where lambda_c = ln(omega_c / (1 - omega_c)).
    """
    logits = compute_pbo_logits(is_performance, oos_performance)
    if logits.size == 0:
        return 1.0
    pbo_val = float(np.mean(logits <= 0.0))
    return float(min(max(pbo_val, 0.0), 1.0))


def evaluate_challenger_gates(
    model_id: str,
    oos_net_returns: NDArray[np.float64] | Sequence[float],
    benchmark_returns: NDArray[np.float64] | Sequence[float] | None = None,
    is_split_metrics: NDArray[np.float64] | Sequence[float] | None = None,
    oos_split_metrics: NDArray[np.float64] | Sequence[float] | None = None,
    expected_net_pnl_bps: float | None = None,
    positions: NDArray[np.float64] | None = None,
    purge_bars: int = 16,
    embargo_pct: float = DEFAULT_EMBARGO_PCT,
    total_trials: int = 45,
    n_groups: int = DEFAULT_N_GROUPS,
    k_test_groups: int = DEFAULT_K_TEST_GROUPS,
) -> CPCVValidationResult:
    """Evaluate all 4 hard net-of-fee promotion gates and return a validated CPCVValidationResult.

    Hard Promotion Gates:
    1. PSR(SR* = 0) >= 0.95 AND PSR(SR* = SR_benchmark) >= 0.95
    2. DSR >= 0.95 (adjusted for total_trials, skewness, and kurtosis)
    3. PBO < 0.30 across the 45 IS/OOS rank logits
    4. E[Net PnL per trade] > 15.0 bps
    """
    if purge_bars < 1:
        raise ValueError(f"purge_bars must be >= 1, got {purge_bars}")
    if total_trials < 1:
        raise ValueError(f"total_trials must be >= 1, got {total_trials}")

    oos_arr = np.asarray(oos_net_returns, dtype=np.float64)
    if oos_arr.ndim == 2:
        # Mean across the phi=9 reconstructed out-of-sample backtest paths
        flat_oos = np.mean(oos_arr, axis=0)
    else:
        flat_oos = oos_arr.ravel()

    sr_net, skewness, kurtosis_pearson = _compute_sample_moments(flat_oos)

    psr_zero = compute_psr(flat_oos, sr_benchmark=0.0)

    if benchmark_returns is not None:
        bench_arr = np.asarray(benchmark_returns, dtype=np.float64).ravel()
        sr_bench, _, _ = _compute_sample_moments(bench_arr)
    else:
        sr_bench = 0.0
    psr_bench = compute_psr(flat_oos, sr_benchmark=sr_bench)

    dsr_val = compute_dsr(
        flat_oos,
        total_trials=total_trials,
        trial_sharpes=oos_split_metrics if (oos_split_metrics is not None and np.asarray(oos_split_metrics).ndim == 1) else None,
    )

    num_splits = math.comb(n_groups, k_test_groups)
    n_oos_paths = math.comb(n_groups - 1, k_test_groups - 1)

    if is_split_metrics is not None and oos_split_metrics is not None:
        pbo_val = compute_pbo(is_split_metrics, oos_split_metrics)
    else:
        # Partition flat_oos into num_splits segments if split metrics were not explicitly provided
        chunks = np.array_split(flat_oos, min(num_splits, max(1, flat_oos.size // 4)))
        chunk_srs = np.array([_compute_sample_moments(c)[0] for c in chunks], dtype=np.float64)
        is_proxy = np.full_like(chunk_srs, max(sr_net, 1e-4))
        pbo_val = compute_pbo(is_proxy, chunk_srs)

    if expected_net_pnl_bps is not None:
        net_pnl_bps = float(expected_net_pnl_bps)
    else:
        net_pnl_bps, _ = _compute_expected_trade_net_pnl_bps(flat_oos, positions)

    return CPCVValidationResult(
        model_id=model_id,
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        num_splits=num_splits,
        n_oos_paths=n_oos_paths,
        purge_bars=purge_bars,
        embargo_pct=embargo_pct,
        psr_vs_zero=psr_zero,
        psr_vs_benchmark=psr_bench,
        dsr=dsr_val,
        pbo=pbo_val,
        expected_net_pnl_bps=net_pnl_bps,
        sharpe_net=sr_net,
        skewness=skewness,
        kurtosis=kurtosis_pearson,
        total_trials=total_trials,
    )


def _evaluate_single_split_worker(
    args: tuple[int, NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
) -> tuple[int, float, float, NDArray[np.float64]]:
    """Top-level picklable worker function for spawned ProcessPoolExecutor CPCV evaluation."""
    split_id, train_idx, test_idx, net_returns = args
    train_ret = net_returns[train_idx]
    test_ret = net_returns[test_idx]
    is_sr, _, _ = _compute_sample_moments(train_ret)
    oos_sr, _, _ = _compute_sample_moments(test_ret)
    return split_id, is_sr, oos_sr, test_ret


class CombinatorialPurgedCV:
    """45-Split Combinatorial Purged Cross-Validation (N=10, k=2, phi=9 OOS paths).

    Enforces:
    - Exact purging of `purge_bars = vertical_barrier_bars` prior to each test block and
      post-test label horizon + `1% embargo` after each test block so zero serial overlap
      occurs between training labels and test labels.
    - Reconstruction of all phi = 9 complete out-of-sample backtest paths from the 45 splits.
    - Out-of-process parallel worker cap of `max_workers <= 2` (2 * 4 = 8 threads max).
    """

    def __init__(
        self,
        vertical_barrier_bars: int = 16,
        n_groups: int = DEFAULT_N_GROUPS,
        k_test_groups: int = DEFAULT_K_TEST_GROUPS,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
        max_workers: int = MAX_CPCV_WORKERS,
    ) -> None:
        if vertical_barrier_bars < 1:
            raise ValueError(f"vertical_barrier_bars (purge_bars) must be >= 1, got {vertical_barrier_bars}")
        if n_groups < 3:
            raise ValueError(f"n_groups must be >= 3, got {n_groups}")
        if not (1 <= k_test_groups < n_groups):
            raise ValueError(f"k_test_groups must be in [1, {n_groups - 1}], got {k_test_groups}")
        if not (0.0 <= embargo_pct < 0.25):
            raise ValueError(f"embargo_pct must be in [0.0, 0.25), got {embargo_pct}")
        if not (1 <= max_workers <= MAX_CPCV_WORKERS):
            raise ValueError(
                f"max_workers must be in [1, {MAX_CPCV_WORKERS}] to respect the 8 P-core ceiling, got {max_workers}"
            )

        self.vertical_barrier_bars: int = vertical_barrier_bars
        self.purge_bars: int = vertical_barrier_bars
        self.n_groups: int = n_groups
        self.k_test_groups: int = k_test_groups
        self.embargo_pct: float = embargo_pct
        self.max_workers: int = max_workers
        self.num_splits: int = math.comb(n_groups, k_test_groups)
        self.n_oos_paths: int = math.comb(n_groups - 1, k_test_groups - 1)

    def compute_embargo_bars(self, n_samples: int) -> int:
        """Compute number of embargo bars after each test block (1% of total bars)."""
        if self.embargo_pct <= 0.0:
            return 0
        return max(1, int(math.ceil(float(n_samples) * self.embargo_pct)))

    def generate_splits(
        self,
        n_samples: int,
        label_end_indices: NDArray[np.int64] | None = None,
    ) -> list[CPCVSplit]:
        """Generate all 45 purged and embargoed train/test splits and their OOS path mappings.

        Guarantees zero serial overlap between training Triple-Barrier label intervals
        [i, i + purge_bars] and test Triple-Barrier label intervals [j, j + purge_bars],
        plus a 1% post-test embargo buffer.
        """
        min_required = self.n_groups * (self.purge_bars + 4)
        if n_samples < min_required:
            raise ValueError(
                f"n_samples ({n_samples}) is too small for {self.n_groups} groups with purge_bars={self.purge_bars} "
                f"(minimum recommended: {min_required})"
            )

        all_indices = np.arange(n_samples, dtype=np.int64)
        group_arrays: list[NDArray[np.int64]] = [
            np.asarray(g, dtype=np.int64) for g in np.array_split(all_indices, self.n_groups)
        ]
        embargo_bars = self.compute_embargo_bars(n_samples)

        if label_end_indices is not None:
            end_idx = np.asarray(label_end_indices, dtype=np.int64).ravel()
            if end_idx.shape[0] != n_samples:
                raise ValueError(
                    f"label_end_indices length ({end_idx.shape[0]}) must match n_samples ({n_samples})"
                )
        else:
            end_idx = np.minimum(all_indices + self.purge_bars, n_samples - 1)

        group_appearance_counter: dict[int, int] = {g: 0 for g in range(self.n_groups)}
        splits: list[CPCVSplit] = []

        for split_id, test_group_tuple in enumerate(combinations(range(self.n_groups), self.k_test_groups)):
            test_group_set = set(test_group_tuple)
            train_group_tuple = tuple(g for g in range(self.n_groups) if g not in test_group_set)

            group_test_slices: dict[int, NDArray[np.int64]] = {}
            oos_path_map: dict[int, int] = {}
            for g in test_group_tuple:
                group_test_slices[g] = group_arrays[g]
                oos_path_map[g] = group_appearance_counter[g]
                group_appearance_counter[g] += 1

            test_indices = np.concatenate([group_arrays[g] for g in test_group_tuple])

            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[test_indices] = True

            purge_mask = np.zeros(n_samples, dtype=bool)
            embargo_mask = np.zeros(n_samples, dtype=bool)

            for g in test_group_tuple:
                g_indices = group_arrays[g]
                t_start = int(g_indices[0])
                t_end = int(g_indices[-1])
                test_label_end = int(np.max(end_idx[g_indices]))

                # 1. Pre-test purge: any training sample prior to t_start whose label reaches >= t_start
                pre_purge_start = max(0, t_start - self.purge_bars)
                if pre_purge_start < t_start:
                    purge_mask[pre_purge_start:t_start] = True
                overlapping_pre = (all_indices < t_start) & (end_idx >= t_start)
                purge_mask |= overlapping_pre

                # 2. Post-test purge + 1% embargo: prevent test labels at t_end from overlapping post-test train bars
                post_purge_end = min(n_samples, max(t_end + 1 + self.purge_bars, test_label_end + 1))
                if t_end + 1 < post_purge_end:
                    purge_mask[t_end + 1 : post_purge_end] = True

                embargo_start = post_purge_end
                embargo_stop = min(n_samples, embargo_start + embargo_bars)
                if embargo_start < embargo_stop:
                    embargo_mask[embargo_start:embargo_stop] = True

            purge_mask &= ~test_mask
            embargo_mask &= ~(test_mask | purge_mask)
            train_mask = ~(test_mask | purge_mask | embargo_mask)

            splits.append(
                CPCVSplit(
                    split_id=split_id,
                    test_groups=test_group_tuple,
                    train_groups=train_group_tuple,
                    train_indices=all_indices[train_mask],
                    test_indices=test_indices,
                    purged_indices=all_indices[purge_mask],
                    embargoed_indices=all_indices[embargo_mask],
                    group_test_slices=group_test_slices,
                    oos_path_map=oos_path_map,
                )
            )

        return splits

    def verify_zero_label_overlap(
        self,
        splits: Sequence[CPCVSplit],
        label_horizon_bars: int | None = None,
    ) -> bool:
        """Verify that no training label interval [i, i + H] overlaps any test label interval [j, j + H]."""
        horizon = label_horizon_bars if label_horizon_bars is not None else self.purge_bars
        for split in splits:
            if split.train_indices.size == 0 or split.test_indices.size == 0:
                return False
            for g_indices in split.group_test_slices.values():
                t_start = int(g_indices[0])
                t_end = int(g_indices[-1])
                # Train bars before test block must satisfy i + horizon < t_start
                pre_train = split.train_indices[split.train_indices < t_start]
                if pre_train.size > 0 and int(np.max(pre_train)) + horizon >= t_start:
                    return False
                # Train bars after test block must satisfy i > t_end + horizon
                post_train = split.train_indices[split.train_indices > t_end]
                if post_train.size > 0 and int(np.min(post_train)) <= t_end + horizon:
                    return False
        return True

    def reconstruct_oos_paths(
        self,
        split_test_returns: dict[int, NDArray[np.float64]] | Sequence[NDArray[np.float64]],
        n_samples: int,
        splits: Sequence[CPCVSplit] | None = None,
    ) -> NDArray[np.float64]:
        """Reconstruct the phi = 9 complete out-of-sample backtest paths of shape (9, n_samples) from 45 splits."""
        active_splits = list(splits) if splits is not None else self.generate_splits(n_samples)
        if len(active_splits) != self.num_splits:
            raise ValueError(f"Expected {self.num_splits} splits, got {len(active_splits)}")

        oos_paths = np.zeros((self.n_oos_paths, n_samples), dtype=np.float64)
        covered = np.zeros((self.n_oos_paths, n_samples), dtype=bool)

        for split in active_splits:
            s_id = split.split_id
            raw_ret = (
                split_test_returns[s_id]
                if isinstance(split_test_returns, dict)
                else split_test_returns[s_id]
            )
            ret_arr = np.asarray(raw_ret, dtype=np.float64).ravel()

            if ret_arr.shape[0] == n_samples:
                for g, g_idx in split.group_test_slices.items():
                    path_idx = split.oos_path_map[g]
                    oos_paths[path_idx, g_idx] = ret_arr[g_idx]
                    covered[path_idx, g_idx] = True
            elif ret_arr.shape[0] == split.test_indices.shape[0]:
                offset = 0
                for g in split.test_groups:
                    g_idx = split.group_test_slices[g]
                    g_len = g_idx.shape[0]
                    path_idx = split.oos_path_map[g]
                    oos_paths[path_idx, g_idx] = ret_arr[offset : offset + g_len]
                    covered[path_idx, g_idx] = True
                    offset += g_len
            else:
                raise ValueError(
                    f"Split {s_id} return length {ret_arr.shape[0]} matches neither "
                    f"test_indices length ({split.test_indices.shape[0]}) nor n_samples ({n_samples})"
                )

        if not np.all(covered):
            missing = int(np.sum(~covered))
            raise RuntimeError(f"Incomplete OOS path reconstruction: {missing} path-bar entries uncovered")

        return oos_paths

    def run_validation(
        self,
        model_id: str,
        gross_asset_returns: NDArray[np.float64],
        positions: NDArray[np.float64],
        funding_rates_8h: NDArray[np.float64] | float = 0.0,
        benchmark_returns: NDArray[np.float64] | None = None,
        fee_schedule: FeeSchedule | None = None,
        bars_per_8h: float = 32.0,
        taker_fill_ratio: float = 0.0,
        total_trials: int = 45,
        parallel: bool = False,
        split_evaluator: Callable[[CPCVSplit, NDArray[np.float64]], tuple[float, float, NDArray[np.float64]]]
        | None = None,
    ) -> tuple[CPCVValidationResult, NDArray[np.float64], list[CPCVSplit]]:
        """Execute full 45-split net-of-fee CPCV, reconstruct 9 OOS paths, and evaluate promotion gates."""
        breakdown = compute_net_of_fee_returns(
            gross_asset_returns=gross_asset_returns,
            positions=positions,
            funding_rates_8h=funding_rates_8h,
            fee_schedule=fee_schedule,
            bars_per_8h=bars_per_8h,
            taker_fill_ratio=taker_fill_ratio,
        )
        net_returns = breakdown.net_returns
        n_samples = net_returns.shape[0]
        splits = self.generate_splits(n_samples)

        is_sharpes = np.empty(self.num_splits, dtype=np.float64)
        oos_sharpes = np.empty(self.num_splits, dtype=np.float64)
        split_test_returns: dict[int, NDArray[np.float64]] = {}

        if split_evaluator is not None:
            for split in splits:
                is_sr, oos_sr, test_ret = split_evaluator(split, net_returns)
                is_sharpes[split.split_id] = is_sr
                oos_sharpes[split.split_id] = oos_sr
                split_test_returns[split.split_id] = test_ret
        elif parallel and self.max_workers > 1:
            mp_ctx = multiprocessing.get_context("spawn")
            tasks = [(s.split_id, s.train_indices, s.test_indices, net_returns) for s in splits]
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp_ctx,
                initializer=_init_cpcv_worker,
                initargs=(OMP_THREADS_PER_WORKER,),
            ) as pool:
                for split_id, is_sr, oos_sr, test_ret in pool.map(_evaluate_single_split_worker, tasks):
                    is_sharpes[split_id] = is_sr
                    oos_sharpes[split_id] = oos_sr
                    split_test_returns[split_id] = test_ret
        else:
            for split in splits:
                s_id, is_sr, oos_sr, test_ret = _evaluate_single_split_worker(
                    (split.split_id, split.train_indices, split.test_indices, net_returns)
                )
                is_sharpes[s_id] = is_sr
                oos_sharpes[s_id] = oos_sr
                split_test_returns[s_id] = test_ret

        oos_paths = self.reconstruct_oos_paths(
            split_test_returns=split_test_returns,
            n_samples=n_samples,
            splits=splits,
        )

        result = evaluate_challenger_gates(
            model_id=model_id,
            oos_net_returns=oos_paths,
            benchmark_returns=benchmark_returns,
            is_split_metrics=is_sharpes,
            oos_split_metrics=oos_sharpes,
            expected_net_pnl_bps=breakdown.expected_net_pnl_bps,
            positions=positions,
            purge_bars=self.purge_bars,
            embargo_pct=self.embargo_pct,
            total_trials=total_trials,
            n_groups=self.n_groups,
            k_test_groups=self.k_test_groups,
        )
        return result, oos_paths, splits


class ChampionChallengerGatekeeper:
    """Champion/Challenger Fallback State Machine for an asset sleeve.

    Rules:
    1. If a newly retrained Challenger model fails ANY of the 4 CPCV promotion gates
       (including PSR < 0.95), discard it, increment `consecutive_failed_retrains`, and
       retain the current Champion model (or transition to `ExecutionRegimeMode.CARRY_ONLY_FALLBACK`
       if no Champion exists or `consecutive_failed_retrains >= 3`).
    2. If the active Champion's rolling 30-day live PSR drops below 0.80 or 3 consecutive
       retrains fail validation, automatically transition that asset sleeve into
       `ExecutionRegimeMode.CARRY_ONLY_FALLBACK` (100% Cash / Delta-Neutral Funding Carry mode)
       until a new Challenger passes all 4 gates.
    """

    def __init__(
        self,
        asset_id: str = "BTCUSDT-PERP",
        initial_champion_id: str | None = None,
        initial_live_30d_psr: float = 1.0,
        initial_state: ChampionChallengerState | None = None,
        min_live_30d_psr: float = LIVE_30D_PSR_FALLBACK_THRESHOLD,
        max_consecutive_failed_retrains: int = MAX_CONSECUTIVE_FAILED_RETRAINS,
    ) -> None:
        self.min_live_30d_psr: float = min_live_30d_psr
        self.max_consecutive_failed_retrains: int = max_consecutive_failed_retrains

        if initial_state is not None:
            self._state = initial_state
        else:
            clamped_psr = float(min(max(initial_live_30d_psr, 0.0), 1.0)) if initial_champion_id else 0.0
            if initial_champion_id is not None and clamped_psr >= self.min_live_30d_psr:
                mode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
                reason = f"Initialized with active Champion '{initial_champion_id}' (30d live PSR={clamped_psr:.3f})."
            else:
                mode = ExecutionRegimeMode.CARRY_ONLY_FALLBACK
                reason = (
                    "Initialized in CARRY_ONLY_FALLBACK mode (no active Champion meeting live PSR threshold)."
                )
            self._state = ChampionChallengerState(
                asset_id=asset_id,
                champion_model_id=initial_champion_id,
                champion_live_30d_psr=clamped_psr,
                consecutive_failed_retrains=0,
                regime_mode=mode,
                last_promotion_reason=reason,
            )

    @property
    def state(self) -> ChampionChallengerState:
        """Current immutable ChampionChallengerState snapshot."""
        return self._state

    @staticmethod
    def _format_failed_gates(result: CPCVValidationResult) -> list[str]:
        """List human-readable descriptions of any failed CPCV promotion gates."""
        reasons: list[str] = []
        if result.psr_vs_zero < PSR_PROMOTION_THRESHOLD:
            reasons.append(f"psr_vs_zero={result.psr_vs_zero:.3f} < {PSR_PROMOTION_THRESHOLD}")
        if result.psr_vs_benchmark < PSR_PROMOTION_THRESHOLD:
            reasons.append(f"psr_vs_benchmark={result.psr_vs_benchmark:.3f} < {PSR_PROMOTION_THRESHOLD}")
        if result.dsr < DSR_PROMOTION_THRESHOLD:
            reasons.append(f"dsr={result.dsr:.3f} < {DSR_PROMOTION_THRESHOLD}")
        if result.pbo >= PBO_MAX_THRESHOLD:
            reasons.append(f"pbo={result.pbo:.3f} >= {PBO_MAX_THRESHOLD}")
        if result.expected_net_pnl_bps <= MIN_EXPECTED_NET_PNL_BPS:
            reasons.append(
                f"expected_net_pnl_bps={result.expected_net_pnl_bps:.2f} <= {MIN_EXPECTED_NET_PNL_BPS}"
            )
        return reasons

    def evaluate_challenger(
        self,
        validation_result: CPCVValidationResult,
        live_psr_on_promotion: float | None = None,
        *,
        sample_bars: int | None = None,
        bars_per_year: float = 365.0 * 96.0,
        target_sharpe_annual: float = 1.0,
    ) -> ChampionChallengerState:
        """Evaluate a newly trained Challenger's CPCVValidationResult and update sleeve state.

        When ``sample_bars`` is given, promotion is refused if the sample is shorter than the
        minimum backtest length for ``validation_result.total_trials`` trials at
        ``target_sharpe_annual`` (REINFORCEMENT_PLAN P1.6, MinBTL).
        """
        min_btl_reason: str | None = None
        if sample_bars is not None:
            from research.trial_registry import passes_min_backtest_length

            ok, sample_years, min_years = passes_min_backtest_length(
                sample_bars, validation_result.total_trials, target_sharpe_annual, bars_per_year
            )
            if not ok:
                min_btl_reason = (
                    f"sample={sample_years:.2f}y < MinBTL={min_years:.2f}y for "
                    f"{validation_result.total_trials} trials at SR*={target_sharpe_annual:.2f}"
                )

        if validation_result.passed_all_gates and min_btl_reason is None:
            new_live_psr = (
                float(min(max(live_psr_on_promotion, 0.0), 1.0))
                if live_psr_on_promotion is not None
                else validation_result.psr_vs_zero
            )
            self._state = ChampionChallengerState(
                asset_id=self._state.asset_id,
                champion_model_id=validation_result.model_id,
                champion_live_30d_psr=new_live_psr,
                consecutive_failed_retrains=0,
                regime_mode=ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY,
                last_promotion_reason=(
                    f"Promoted Challenger '{validation_result.model_id}' to Champion after passing all 4 "
                    f"net-of-fee CPCV gates (PSR_0={validation_result.psr_vs_zero:.3f}, "
                    f"PSR_bench={validation_result.psr_vs_benchmark:.3f}, DSR={validation_result.dsr:.3f}, "
                    f"PBO={validation_result.pbo:.3f}, E[NetPnL]={validation_result.expected_net_pnl_bps:.2f} bps)."
                ),
            )
            return self._state

        # Challenger failed at least one gate (or the MinBTL rule) -> discard Challenger
        failed = self._format_failed_gates(validation_result)
        if min_btl_reason is not None:
            failed.append(min_btl_reason)
        failed_gates_str = ", ".join(failed)
        new_failed_count = self._state.consecutive_failed_retrains + 1

        should_fallback = (
            self._state.champion_model_id is None
            or new_failed_count >= self.max_consecutive_failed_retrains
            or self._state.champion_live_30d_psr < self.min_live_30d_psr
            or self._state.regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK
        )

        if should_fallback:
            new_mode = ExecutionRegimeMode.CARRY_ONLY_FALLBACK
            fallback_detail = (
                f"Transitioned to CARRY_ONLY_FALLBACK (consecutive_failed_retrains={new_failed_count}, "
                f"champion='{self._state.champion_model_id}', live_30d_psr={self._state.champion_live_30d_psr:.3f})."
            )
        else:
            new_mode = self._state.regime_mode
            fallback_detail = (
                f"Retained current Champion '{self._state.champion_model_id}' "
                f"(consecutive_failed_retrains={new_failed_count}/{self.max_consecutive_failed_retrains})."
            )

        self._state = ChampionChallengerState(
            asset_id=self._state.asset_id,
            champion_model_id=self._state.champion_model_id,
            champion_live_30d_psr=self._state.champion_live_30d_psr,
            consecutive_failed_retrains=new_failed_count,
            regime_mode=new_mode,
            last_promotion_reason=(
                f"Discarded Challenger '{validation_result.model_id}' due to failed gates [{failed_gates_str}]. "
                f"{fallback_detail}"
            ),
        )
        return self._state

    def update_live_30d_psr(
        self,
        live_30d_psr: float | None = None,
        live_30d_net_returns: NDArray[np.float64] | Sequence[float] | None = None,
        sr_benchmark: float = 0.0,
    ) -> ChampionChallengerState:
        """Update the active Champion's rolling 30-day live PSR and trigger CARRY_ONLY_FALLBACK if < 0.80."""
        if live_30d_psr is not None:
            psr_val = float(min(max(live_30d_psr, 0.0), 1.0))
        elif live_30d_net_returns is not None:
            psr_val = compute_psr(live_30d_net_returns, sr_benchmark=sr_benchmark)
        else:
            raise ValueError("Either live_30d_psr or live_30d_net_returns must be provided.")

        if (
            self._state.champion_model_id is None
            or psr_val < self.min_live_30d_psr
            or self._state.consecutive_failed_retrains >= self.max_consecutive_failed_retrains
            or self._state.regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK
        ):
            new_mode = ExecutionRegimeMode.CARRY_ONLY_FALLBACK
            reason = (
                f"Sleeve '{self._state.asset_id}' in CARRY_ONLY_FALLBACK: "
                f"rolling 30d live PSR={psr_val:.3f} (threshold={self.min_live_30d_psr:.2f}), "
                f"consecutive_failed_retrains={self._state.consecutive_failed_retrains}. "
                "Awaiting new Challenger passing all 4 CPCV gates."
            )
        else:
            new_mode = self._state.regime_mode
            reason = (
                f"Champion '{self._state.champion_model_id}' healthy with rolling 30d live PSR={psr_val:.3f} "
                f">= {self.min_live_30d_psr:.2f}."
            )

        self._state = ChampionChallengerState(
            asset_id=self._state.asset_id,
            champion_model_id=self._state.champion_model_id,
            champion_live_30d_psr=psr_val,
            consecutive_failed_retrains=self._state.consecutive_failed_retrains,
            regime_mode=new_mode,
            last_promotion_reason=reason,
        )
        return self._state
